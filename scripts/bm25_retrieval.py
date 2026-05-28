#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
}


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


def tokenize(text, remove_stopwords=True):
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def concept_text(row):
    return f"{row.get('term', '')} {row.get('definition', '')}"


def query_text(row):
    document_type = row.get("document_type") or ""
    return f"{document_type} {row.get('text', '')}"


class BM25Index:
    def __init__(self, concept_rows, k1=1.2, b=0.75, remove_stopwords=True):
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.doc_ids = []
        self.doc_lengths = {}
        self.inverted = defaultdict(list)
        self.idf = {}
        self.average_doc_length = 0.0
        self._build(concept_rows)

    def _build(self, concept_rows):
        document_frequencies = Counter()
        total_length = 0

        for row in concept_rows:
            concept_id = row["concept_id"]
            tokens = tokenize(concept_text(row), remove_stopwords=self.remove_stopwords)
            term_frequencies = Counter(tokens)
            doc_length = len(tokens) or 1

            self.doc_ids.append(concept_id)
            self.doc_lengths[concept_id] = doc_length
            total_length += doc_length

            for token, frequency in term_frequencies.items():
                self.inverted[token].append((concept_id, frequency))
                document_frequencies[token] += 1

        doc_count = len(self.doc_ids)
        self.average_doc_length = total_length / doc_count if doc_count else 0.0
        for token, frequency in document_frequencies.items():
            self.idf[token] = math.log(1.0 + (doc_count - frequency + 0.5) / (frequency + 0.5))

    def search(self, text, top_k=100):
        query_terms = Counter(tokenize(text, remove_stopwords=self.remove_stopwords))
        scores = defaultdict(float)
        avgdl = self.average_doc_length or 1.0

        for token, query_frequency in query_terms.items():
            idf = self.idf.get(token)
            if idf is None:
                continue
            for concept_id, term_frequency in self.inverted[token]:
                doc_length = self.doc_lengths[concept_id]
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * doc_length / avgdl)
                scores[concept_id] += query_frequency * idf * (term_frequency * (self.k1 + 1.0) / denominator)

        if not scores:
            return self.doc_ids[:top_k]

        ranked_ids = sorted(scores, key=lambda concept_id: (-scores[concept_id], concept_id))
        if len(ranked_ids) < top_k:
            seen = set(ranked_ids)
            ranked_ids.extend(concept_id for concept_id in self.doc_ids if concept_id not in seen)
        return ranked_ids[:top_k]


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


def compute_metrics(rankings, val_rows):
    mrr = []
    recall_5 = []
    recall_10 = []
    ndcg_10 = []
    for row in val_rows:
        relevant_ids = set(row["retrieval_labels"]["positive_ids"])
        ranked_ids = rankings[row["id"]]
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


def rank_rows(index, rows, top_k):
    return {row["id"]: index.search(query_text(row), top_k=top_k) for row in rows}


def ranking_rows(rankings):
    return [{"id": query_id, "ranked_ids": ranked_ids} for query_id, ranked_ids in rankings.items()]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Track1 BM25 retrieval rankings.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--keep-stopwords", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    concept_rows = read_jsonl(dataset_root / "concept_pool.jsonl")
    val_rows = read_jsonl(dataset_root / "val.jsonl")
    test_rows = read_jsonl(dataset_root / "test_input.jsonl")

    index = BM25Index(
        concept_rows,
        k1=args.k1,
        b=args.b,
        remove_stopwords=not args.keep_stopwords,
    )
    val_rankings = rank_rows(index, val_rows, top_k=args.top_k)
    test_rankings = rank_rows(index, test_rows, top_k=args.top_k)
    metrics = compute_metrics(val_rankings, val_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "val_retrieval_ranking.jsonl", ranking_rows(val_rankings))
    write_jsonl(output_dir / "test_retrieval_ranking.jsonl", ranking_rows(test_rankings))
    write_json(
        output_dir / "metrics.json",
        {
            **metrics,
            "bm25": {
                "b": args.b,
                "k1": args.k1,
                "remove_stopwords": not args.keep_stopwords,
                "top_k": args.top_k,
            },
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
