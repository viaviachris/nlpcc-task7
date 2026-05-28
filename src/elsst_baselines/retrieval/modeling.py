from pathlib import Path

from elsst_baselines.common.lora import discover_lora_target_modules


def _torch_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _resolve_local_hf_snapshot(model_name):
    if Path(model_name).exists():
        return model_name

    if "/" not in model_name:
        return model_name

    namespace, repo = model_name.split("/", 1)
    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_root = hub_root / f"models--{namespace}--{repo}"
    snapshots_dir = model_root / "snapshots"
    if not snapshots_dir.exists():
        return model_name

    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return model_name
    return str(snapshots[-1])


def _load_sentence_transformer(model_name, max_seq_length):
    from sentence_transformers import SentenceTransformer

    model_name = _resolve_local_hf_snapshot(model_name)
    model_kwargs = {"device_map": "auto"}
    processor_kwargs = {"padding_side": "left"}
    try:
        import flash_attn  # noqa: F401

        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    model = SentenceTransformer(
        model_name,
        model_kwargs=model_kwargs,
        processor_kwargs=processor_kwargs,
    )
    first_module = model._first_module()
    inferred_limit = None
    if hasattr(first_module, "auto_model") and hasattr(first_module.auto_model, "config"):
        inferred_limit = getattr(first_module.auto_model.config, "max_position_embeddings", None)
    if isinstance(inferred_limit, int) and inferred_limit > 0:
        model.max_seq_length = min(max_seq_length, inferred_limit)
    else:
        model.max_seq_length = max_seq_length
    return model


def _is_sentence_transformer_checkpoint(path):
    path = Path(path)
    return path.is_dir() and (path / "modules.json").exists() and (path / "model.safetensors").exists()


def _is_peft_adapter_dir(path):
    path = Path(path)
    return path.is_dir() and (path / "adapter_config.json").exists()


def _checkpoint_uses_lora(path):
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(str(path / "model.safetensors"))
    return any(".lora_A." in key or ".lora_B." in key for key in state.keys())


def _checkpoint_base_model_name(path):
    import json

    path = Path(path)
    readme_path = path / "README.md"
    if readme_path.exists():
        for line in readme_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("base_model:"):
                return line.split(":", 1)[1].strip()

    for candidate in [path / "base_model_name.txt", path.parent / "base_model_name.txt"]:
        if candidate.exists():
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                return value

    config_path = path / "config.json"
    tokenizer_config_path = path / "tokenizer_config.json"
    if config_path.exists() and tokenizer_config_path.exists():
        checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint_tokenizer = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        hub_root = Path.home() / ".cache" / "huggingface" / "hub"
        snapshots = sorted(hub_root.glob("models--*--*/snapshots/*"))
        matches = []
        for snapshot in snapshots:
            snapshot_config = snapshot / "config.json"
            snapshot_tokenizer = snapshot / "tokenizer_config.json"
            if not snapshot_config.exists() or not snapshot_tokenizer.exists():
                continue
            try:
                candidate_config = json.loads(snapshot_config.read_text(encoding="utf-8"))
                candidate_tokenizer = json.loads(snapshot_tokenizer.read_text(encoding="utf-8"))
            except Exception:
                continue
            keys = [
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
                "max_position_embeddings",
                "vocab_size",
            ]
            if any(checkpoint_config.get(key) != candidate_config.get(key) for key in keys):
                continue
            if checkpoint_tokenizer.get("tokenizer_class") != candidate_tokenizer.get("tokenizer_class"):
                continue
            matches.append(str(snapshot))
        if len(matches) == 1:
            return matches[0]

    raise RuntimeError(f"could not determine base model name from checkpoint at {path}")


def _load_sentence_transformer_checkpoint_with_lora(path, max_seq_length):
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors.torch import load_file

    base_model_name = _checkpoint_base_model_name(path)
    model = _load_sentence_transformer(base_model_name, max_seq_length=max_seq_length)
    first_module, backbone = _sentence_transformer_backbone(model)
    target_modules = discover_lora_target_modules(backbone)

    peft_task_type = getattr(TaskType, "FEATURE_EXTRACTION", None)
    if peft_task_type is None:
        raise RuntimeError("installed PEFT does not expose TaskType.FEATURE_EXTRACTION")

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=peft_task_type,
        target_modules=target_modules,
    )
    first_module.auto_model = get_peft_model(backbone, config)
    state = load_file(str(Path(path) / "model.safetensors"))
    incompatible = first_module.auto_model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "failed to restore LoRA checkpoint cleanly: "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
    return model


def _sentence_transformer_backbone(model):
    first_module = model._first_module()
    if hasattr(first_module, "auto_model"):
        return first_module, first_module.auto_model
    raise RuntimeError("could not locate the transformer backbone inside SentenceTransformer")


def load_retrieval_train_bundle(model_name, max_seq_length):
    from peft import LoraConfig, TaskType, get_peft_model

    model = _load_sentence_transformer(model_name, max_seq_length=max_seq_length)
    first_module, backbone = _sentence_transformer_backbone(model)
    target_modules = discover_lora_target_modules(backbone)

    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False

    peft_task_type = getattr(TaskType, "FEATURE_EXTRACTION", None)
    if peft_task_type is None:
        raise RuntimeError("installed PEFT does not expose TaskType.FEATURE_EXTRACTION")

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=peft_task_type,
        target_modules=target_modules,
    )
    first_module.auto_model = get_peft_model(backbone, config)
    return model, target_modules


def load_retrieval_inference_model(model_name, max_seq_length, adapter_dir=None):
    from peft import PeftModel

    if _is_sentence_transformer_checkpoint(model_name):
        if _checkpoint_uses_lora(model_name):
            return _load_sentence_transformer_checkpoint_with_lora(model_name, max_seq_length=max_seq_length)
        return _load_sentence_transformer(str(model_name), max_seq_length=max_seq_length)

    model = _load_sentence_transformer(model_name, max_seq_length=max_seq_length)
    if adapter_dir and _is_peft_adapter_dir(adapter_dir):
        first_module, backbone = _sentence_transformer_backbone(model)
        first_module.auto_model = PeftModel.from_pretrained(backbone, str(adapter_dir))
    return model


def save_retrieval_artifacts(model, output_dir, model_name, target_modules, merge_adapter=False):
    output_dir = Path(output_dir)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    first_module, backbone = _sentence_transformer_backbone(model)
    backbone.save_pretrained(adapter_dir)
    first_module.tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "base_model_name.txt").write_text(model_name + "\n", encoding="utf-8")
    (adapter_dir / "target_modules.txt").write_text("\n".join(target_modules) + "\n", encoding="utf-8")

    if merge_adapter and hasattr(backbone, "merge_and_unload"):
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = backbone.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        first_module.tokenizer.save_pretrained(merged_dir)
