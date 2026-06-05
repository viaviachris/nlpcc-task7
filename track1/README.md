---
pretty_name: ELSST Track1
language:
  - en
task_categories:
  - text-retrieval
tags:
  - benchmark
  - information-retrieval
  - social-science
size_categories:
  - 1K<n<10K
---

# ELSST Track1: Implicit Concept Retrieval

ELSST Track1 evaluates whether a model can read a long synthetic social-science passage and retrieve the most relevant concepts from a fixed ELSST concept pool. The target is not lexical matching. The concepts are intentionally implicit, cross-sentence, and often require discourse-level reasoning over topic, framing, and social context.

This card is the authoritative task description for the retrieval track. The published dataset lives at [JohnWang10086/elsst-track1](https://huggingface.co/datasets/JohnWang10086/elsst-track1). The companion generation track is published at [JohnWang10086/elsst-track2](https://huggingface.co/datasets/JohnWang10086/elsst-track2). The reference code lives in [ZeqiangWangAI/elsst-benchmark-baselines](https://github.com/ZeqiangWangAI/elsst-benchmark-baselines).

## Task

Given a passage, rank the ELSST concepts in `concept_pool.jsonl` by relevance.

- Train and validation contain gold concepts and hard negatives.
- `test_input.jsonl` does **not** expose any gold labels.
- Public sample identifiers are release-safe IDs of the form `split_legacy_id`, for example `train_t00023` and `val_v00029`.

## Splits

| Split | File | Rows | Labels exposed |
| --- | --- | ---: | --- |
| train | `train.jsonl` | 2,985 | Yes |
| val | `val.jsonl` | 756 | Yes |
| test | `test_input.jsonl` | 1,911 | No |
| pool | `concept_pool.jsonl` | 3,433 concepts | N/A |

## Schema

`train.jsonl` and `val.jsonl`

```json
{
  "id": "train_t00023",
  "text": "long passage...",
  "document_type": "op_ed",
  "generation_labels": [
    {"concept_id": "...", "term": "...", "definition": "..."}
  ],
  "retrieval_labels": {
    "positive_ids": ["..."],
    "hard_negative_ids": ["..."]
  }
}
```

`test_input.jsonl`

```json
{
  "id": "test_t00009",
  "text": "long passage...",
  "document_type": "research_summary"
}
```

## Evaluation

The baseline retrieval evaluator reports:

- `MRR`
- `Recall@5`
- `Recall@10`
- `NDCG@10`

Evaluation is defined over the validation split only. The public test split is input-only.

## Data Design And Quality

The corpus is fully synthetic. It was generated for benchmark construction and does not contain real personal data.

Document types are intentionally balanced across 10 genres:

- `blog_post`
- `case_study`
- `encyclopedia_entry`
- `forum_discussion`
- `interview_transcript`
- `news_article`
- `op_ed`
- `policy_brief`
- `report_excerpt`
- `research_summary`

Observed passage-length statistics in the current release:

| Split | Min | Median | Max |
| --- | ---: | ---: | ---: |
| train | 629 | 894 | 1,177 |
| val | 672 | 889.5 | 1,130 |
| test | 675 | 896 | 1,154 |

Label-count distributions:

- train: `{1: 605, 2: 598, 3: 579, 4: 613, 5: 590}`
- val: `{1: 153, 2: 153, 3: 152, 4: 155, 5: 143}`
- test gold labels are intentionally withheld

## Release Notes

Release date: March 29, 2026.

- All public IDs were normalized to `train_*`, `val_*`, and `test_*` so Track1 and Track2 share the same stable public ID mapping.

## Baseline Status

The reference retrieval code lives in [ZeqiangWangAI/elsst-benchmark-baselines](https://github.com/ZeqiangWangAI/elsst-benchmark-baselines). The nominal baseline model is `Qwen/Qwen3-Embedding-0.6B`.

## Citation

If you use Track1, cite this benchmark release together with the companion Track2 card so the full task definition is preserved.
# nlpcc-task7
# Track1 检索实验结果总结
## 任务与指标

Track1 是隐式概念检索任务：给定查询或样例文本，从固定的 ELSST 概念池中检索并排序相关概念。本实验主要关注以下指标：

- `MRR`：平均倒数排名，衡量第一个相关结果出现得是否靠前。
- `NDCG@10`：Top 10 排序质量，也是训练阶段主要使用的模型选择指标。
- `Recall@10`：Top 10 中召回相关概念的能力。
- `Recall@5`：更严格的前排召回能力。

## 基线结果

本实验先记录了两个 Track1 LoRA 微调基线：

| 模型 | 方法 | Recall@5 |
|---|---|---:|
| Qwen3-0.6B Embedding | LoRA 微调 | 0.4490 |
| e5-small | LoRA 微调 | 0.3420 |

这两个结果作为早期基线，用于衡量后续 checkpoint 选择、评估修复、融合与检索增强方法带来的提升。

## 实验方法演进

### 1. 单模型 Best 对比

第一阶段先比较不同 embedding 底座在单模型检索下的最佳结果。为了避免把同一训练方案下的不同 checkpoint 误写成不同方法，主表中只保留每个模型系列的最佳结果。

| 模型 | 方法说明 | MRR | NDCG@10 | Recall@10 | Recall@5 |
|---|---|---:|---:|---:|---:|
| E5-small  | Encoder-only embedding，修复评估后选择最佳 checkpoint | 0.5768 | 0.4357 | 0.5074 | 0.4100 |
| E5-base  | 更大规模的 E5 encoder-only embedding，选择最佳 checkpoint | 0.5590 | 0.4273 | 0.5067 | 0.4300 |
| Qwen3-0.6B  | Qwen3-Embedding-0.6B LoRA 微调，选择最佳 checkpoint | 0.6959 | 0.5769 | 0.6612 | 0.5572 |
| Qwen3-8B  | Qwen3-Embedding-8B LoRA 微调，选择最佳 checkpoint | 0.8857 | 0.7760 | 0.8330 |  0.7251 |



