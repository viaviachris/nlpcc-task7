#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

from sentence_transformers import CrossEncoder


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_concepts(path):
    return {
        row["concept_id"]: {
            "term": row["term"],
            "definition": row["definition"],
        }
        for row in read_jsonl(path)
    }


def load_rankings(path):
    return {row["id"]: row["ranked_ids"] for row in read_jsonl(path)}


def concept_text(concept):
    return f"Term: {concept['term']}\nDefinition: {concept['definition']}"


def query_text(row):
    document_type = row.get("document_type") or "unknown"
    return f"Document type: {document_type}\nText: {row['text']}"


def reciprocal_rank(ranked_ids, relevant_ids):
    for index, concept_id in enumerate(ranked_ids, start=1):
        if concept_id in relevant_ids:
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def ndcg_at_k(ranked_ids, relevant_ids, k):
    gains = [1.0 if concept_id in relevant_ids else 0.0 for concept_id in ranked_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    idcg = sum(1.0 / math.log2(index + 2) for index in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg else 0.0


def compute_metrics(rankings, rows):
    mrr = []
    recall_5 = []
    recall_10 = []
    ndcg_10 = []
    for row in rows:
        relevant_ids = set(row["retrieval_labels"]["positive_ids"])
        ranked_ids = rankings[row["id"]]
        mrr.append(reciprocal_rank(ranked_ids, relevant_ids))
        recall_5.append(recall_at_k(ranked_ids, relevant_ids, 5))
        recall_10.append(recall_at_k(ranked_ids, relevant_ids, 10))
        ndcg_10.append(ndcg_at_k(ranked_ids, relevant_ids, 10))
    count = len(rows)
    return {
        "MRR": sum(mrr) / count if count else 0.0,
        "NDCG@10": sum(ndcg_10) / count if count else 0.0,
        "Recall@10": sum(recall_10) / count if count else 0.0,
        "Recall@5": sum(recall_5) / count if count else 0.0,
    }


def rerank_split(model, rows, rankings, concepts, rerank_top_k, batch_size, split_name):
    reranked = {}
    row_map = {row["id"]: row for row in rows}
    total = len(rankings)

    for index, (row_id, ranked_ids) in enumerate(rankings.items(), start=1):
        row = row_map[row_id]
        candidate_ids = ranked_ids[:rerank_top_k]
        pairs = [(query_text(row), concept_text(concepts[concept_id])) for concept_id in candidate_ids]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scored = list(zip(candidate_ids, scores))
        scored.sort(key=lambda item: (-float(item[1]), ranked_ids.index(item[0]), item[0]))
        reranked[row_id] = [concept_id for concept_id, _ in scored] + ranked_ids[rerank_top_k:]
        if index == 1 or index % 25 == 0 or index == total:
            print(f"[{split_name}] reranked {index}/{total}", flush=True)

    return reranked


def ranking_rows(rankings):
    return [{"id": query_id, "ranked_ids": ranked_ids} for query_id, ranked_ids in rankings.items()]


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank Track1 retrieval candidates with a cross-encoder.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--input-dir", required=True, help="Directory containing val/test_retrieval_ranking.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="BAAI/bge-reranker-base")
    parser.add_argument("--rerank-top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    concepts = load_concepts(dataset_root / "concept_pool.jsonl")
    val_rows = read_jsonl(dataset_root / "val.jsonl")
    test_rows = read_jsonl(dataset_root / "test_input.jsonl")
    val_rankings = load_rankings(input_dir / "val_retrieval_ranking.jsonl")
    test_rankings = load_rankings(input_dir / "test_retrieval_ranking.jsonl")

    model = CrossEncoder(args.model_name, max_length=args.max_length, device=args.device)

    val_reranked = rerank_split(
        model,
        val_rows,
        val_rankings,
        concepts,
        rerank_top_k=args.rerank_top_k,
        batch_size=args.batch_size,
        split_name="val",
    )
    test_reranked = rerank_split(
        model,
        test_rows,
        test_rankings,
        concepts,
        rerank_top_k=args.rerank_top_k,
        batch_size=args.batch_size,
        split_name="test",
    )
    metrics = compute_metrics(val_reranked, val_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "val_retrieval_ranking.jsonl", ranking_rows(val_reranked))
    write_jsonl(output_dir / "test_retrieval_ranking.jsonl", ranking_rows(test_reranked))
    write_json(
        output_dir / "metrics.json",
        {
            **metrics,
            "reranker": {
                "batch_size": args.batch_size,
                "input_dir": str(input_dir),
                "max_length": args.max_length,
                "model_name": args.model_name,
                "rerank_top_k": args.rerank_top_k,
            },
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
