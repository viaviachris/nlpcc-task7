from pathlib import Path
import hashlib

from elsst_baselines.common.jsonl import read_jsonl


QUERY_INSTRUCTION = "Instruct: Given a long social-science passage, retrieve the most relevant ELSST concepts."
QUERY_INSTRUCTION_V1 = (
    "query: Retrieve the most relevant ELSST controlled-vocabulary concepts for this social-science passage."
)
QUERY_INSTRUCTION_QWEN3_V1 = (
    "Instruct: Identify the implicit ELSST social-science concepts that best describe the passage. "
    "Use the document type as context and rank controlled-vocabulary concepts by relevance."
)


def format_query(text, document_type=None, prompt_style="baseline"):
    if prompt_style == "qwen3_embedding_v1":
        doc_type = document_type or "unknown"
        return f"{QUERY_INSTRUCTION_QWEN3_V1}\nDocument type: {doc_type}\nQuery: {text}"
    if prompt_style == "retrieval_v1":
        doc_type = document_type or "unknown"
        return f"{QUERY_INSTRUCTION_V1}\nDocument type: {doc_type}\nPassage: {text}"
    return f"{QUERY_INSTRUCTION}\nQuery: {text}"


def format_concept(concept, prompt_style="baseline"):
    if prompt_style == "qwen3_embedding_v1":
        return (
            "Concept document: ELSST controlled-vocabulary entry\n"
            f"Term: {concept['term']}\n"
            f"Definition: {concept['definition']}"
        )
    if prompt_style == "retrieval_v1":
        return f"passage: ELSST concept\nTerm: {concept['term']}\nDefinition: {concept['definition']}"
    return f"Concept: {concept['term']}\nDefinition: {concept['definition']}"


def _sample_negative_ids(hard_negative_ids, limit, sample_key):
    if limit is None or limit <= 0 or len(hard_negative_ids) <= limit:
        return hard_negative_ids

    ranked = sorted(
        hard_negative_ids,
        key=lambda negative_id: hashlib.sha256(f"{sample_key}::{negative_id}".encode("utf-8")).hexdigest(),
    )
    return ranked[:limit]


def load_concept_pool(path):
    rows = read_jsonl(path)
    return {
        row["concept_id"]: {"term": row["term"], "definition": row["definition"]}
        for row in rows
    }


def load_track_rows(path, max_rows=None):
    rows = read_jsonl(path)
    if max_rows is not None:
        return rows[:max_rows]
    return rows


def build_retrieval_triplets(rows, concept_pool, prompt_style="baseline", negatives_per_positive=None):
    triplets = []
    for row in rows:
        query = format_query(
            row["text"],
            document_type=row.get("document_type"),
            prompt_style=prompt_style,
        )
        positive_map = {label["concept_id"]: label for label in row["generation_labels"]}
        for positive_id in row["retrieval_labels"]["positive_ids"]:
            positive_label = positive_map[positive_id]
            positive_text = format_concept(positive_label, prompt_style=prompt_style)
            negative_ids = _sample_negative_ids(
                row["retrieval_labels"]["hard_negative_ids"],
                negatives_per_positive,
                sample_key=f"{row['id']}::{positive_id}",
            )
            for negative_id in negative_ids:
                negative_text = format_concept(concept_pool[negative_id], prompt_style=prompt_style)
                triplets.append(
                    {
                        "query_id": row["id"],
                        "query": query,
                        "positive_id": positive_id,
                        "positive": positive_text,
                        "negative_id": negative_id,
                        "negative": negative_text,
                    }
                )
    return triplets


def build_ir_evaluation_payload(rows, concept_pool, prompt_style="baseline"):
    queries = {}
    relevant_docs = {}
    for row in rows:
        queries[row["id"]] = format_query(
            row["text"],
            document_type=row.get("document_type"),
            prompt_style=prompt_style,
        )
        relevant_docs[row["id"]] = set(row["retrieval_labels"]["positive_ids"])
    corpus = {
        concept_id: format_concept(
            {"term": concept["term"], "definition": concept["definition"]},
            prompt_style=prompt_style,
        )
        for concept_id, concept in concept_pool.items()
    }
    return queries, corpus, relevant_docs


def retrieval_dataset_summary(
    dataset_root,
    max_train_samples=None,
    max_eval_samples=None,
    prompt_style="baseline",
    negatives_per_positive=None,
):
    dataset_root = Path(dataset_root)
    concept_pool = load_concept_pool(dataset_root / "concept_pool.jsonl")
    train_rows = load_track_rows(dataset_root / "train.jsonl")
    val_rows = load_track_rows(dataset_root / "val.jsonl")
    train_triplets = build_retrieval_triplets(
        train_rows,
        concept_pool,
        prompt_style=prompt_style,
        negatives_per_positive=negatives_per_positive,
    )
    eval_rows = val_rows[:max_eval_samples] if max_eval_samples else val_rows
    return {
        "dataset_root": str(dataset_root),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "concept_pool_size": len(concept_pool),
        "train_triplets_total": len(train_triplets),
        "requested_max_train_samples": max_train_samples,
        "requested_max_eval_samples": max_eval_samples,
        "prompt_style": prompt_style,
        "negatives_per_positive": negatives_per_positive,
        "effective_train_samples": min(len(train_triplets), max_train_samples) if max_train_samples else len(train_triplets),
        "effective_eval_queries": len(eval_rows),
    }
