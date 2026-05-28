#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


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


def load_rankings(path):
    return {row["id"]: row["ranked_ids"] for row in read_jsonl(path)}


def fuse_rankings(ranking_sets, rrf_k=60, top_k=100, weights=None):
    weights = weights or [1.0] * len(ranking_sets)
    query_ids = list(ranking_sets[0].keys())
    fused = {}
    for query_id in query_ids:
        scores = defaultdict(float)
        first_seen = {}
        for model_index, rankings in enumerate(ranking_sets):
            ranked_ids = rankings[query_id]
            for rank, concept_id in enumerate(ranked_ids, start=1):
                scores[concept_id] += weights[model_index] / (rrf_k + rank)
                first_seen.setdefault(concept_id, rank)
        sorted_ids = sorted(scores, key=lambda concept_id: (-scores[concept_id], first_seen[concept_id], concept_id))
        fused[query_id] = sorted_ids[:top_k]
    return fused


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
    import math

    gains = [1.0 if concept_id in relevant_ids else 0.0 for concept_id in ranked_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    idcg = sum(1.0 / math.log2(index + 2) for index in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg else 0.0


def compute_metrics(val_rankings, val_path):
    val_rows = read_jsonl(val_path)
    mrr = []
    recall_5 = []
    recall_10 = []
    ndcg_10 = []
    for row in val_rows:
        query_id = row["id"]
        relevant_ids = set(row["retrieval_labels"]["positive_ids"])
        ranked_ids = val_rankings[query_id]
        mrr.append(reciprocal_rank(ranked_ids, relevant_ids))
        recall_5.append(recall_at_k(ranked_ids, relevant_ids, 5))
        recall_10.append(recall_at_k(ranked_ids, relevant_ids, 10))
        ndcg_10.append(ndcg_at_k(ranked_ids, relevant_ids, 10))
    count = len(val_rows)
    return {
        "MRR": sum(mrr) / count if count else 0.0,
        "NDCG@10": sum(ndcg_10) / count if count else 0.0,
        "Recall@10": sum(recall_10) / count if count else 0.0,
        "Recall@5": sum(recall_5) / count if count else 0.0,
    }


def rows_from_rankings(rankings):
    return [{"id": query_id, "ranked_ids": ranked_ids} for query_id, ranked_ids in rankings.items()]


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse Track1 retrieval rankings with reciprocal rank fusion.")
    parser.add_argument("--eval-dir", action="append", required=True, help="Directory containing val/test_retrieval_ranking.jsonl.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-file", default="/home/libo/nlpcc-task7/track1/val.jsonl")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--weights", nargs="*", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    eval_dirs = [Path(path) for path in args.eval_dir]
    if args.weights and len(args.weights) != len(eval_dirs):
        raise SystemExit("--weights length must match the number of --eval-dir arguments")

    val_sets = [load_rankings(eval_dir / "val_retrieval_ranking.jsonl") for eval_dir in eval_dirs]
    test_sets = [load_rankings(eval_dir / "test_retrieval_ranking.jsonl") for eval_dir in eval_dirs]

    val_fused = fuse_rankings(val_sets, rrf_k=args.rrf_k, top_k=args.top_k, weights=args.weights)
    test_fused = fuse_rankings(test_sets, rrf_k=args.rrf_k, top_k=args.top_k, weights=args.weights)
    metrics = compute_metrics(val_fused, args.val_file)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "val_retrieval_ranking.jsonl", rows_from_rankings(val_fused))
    write_jsonl(output_dir / "test_retrieval_ranking.jsonl", rows_from_rankings(test_fused))
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
