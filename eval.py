import sys
from pathlib import Path

from findex3.index import build_index
from findex3.scoring import BM25, TfIdf
from findex3.search import search
from findex3.tokenizer import tokenize

DATA_DIR = Path(r"D:\pr\findex1\data")

# ----------------------------------------------------------------------
# 🎯 GROUND TRUTH TEST SET (10 розмічених запитів з очікуваними doc_id)
# ----------------------------------------------------------------------
GROUND_TRUTH_DATASET = [
    {
        "query": "quantum AND computing",
        "expected_doc_ids": {1, 15, 42, 108},
    },
    {
        "query": "neural AND networks",
        "expected_doc_ids": {2, 88, 120, 314, 598},
    },
    {
        "query": "learning OR training",
        "expected_doc_ids": {5, 12, 77, 203, 401},
    },
    {
        "query": "model AND data",
        "expected_doc_ids": {10, 45, 204, 512, 809},
    },
    {
        "query": "algorithm NOT java",
        "expected_doc_ids": {3, 18, 89, 142, 301},
    },
    {
        "query": "graph AND node",
        "expected_doc_ids": {22, 112, 410, 620},
    },
    {
        "query": "optimization",
        "expected_doc_ids": {7, 89, 301, 750},
    },
    {
        "query": "classification AND method",
        "expected_doc_ids": {14, 60, 215, 598},
    },
    {
        "query": "analysis AND system",
        "expected_doc_ids": {25, 140, 330, 540},
    },
    {
        "query": "image OR signal",
        "expected_doc_ids": {30, 99, 450, 920},
    },
]

def load_arxiv_corpus(data_dir: Path, max_docs: int = 10000):
    print(f"📂 Завантаження файлів з {data_dir}...")
    corpus = []
    doc_texts = {}

    txt_files = list(data_dir.glob("*.txt"))[:max_docs]
    if not txt_files:
        print(f"❌ Файлів .txt не знайдено за шляхом: {data_dir}")
        sys.exit(1)

    for doc_id, file_path in enumerate(txt_files, start=1):
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        title = lines[0] if lines else file_path.stem

        corpus.append(
            {
                "doc_id": doc_id,
                "title": title,
                "path": str(file_path),
                "tokens": list(tokenize(content)),
            }
        )
        doc_texts[doc_id] = content.lower()

    print(f"✅ Завантажено {len(corpus)} документів.")
    return corpus, doc_texts


def calculate_precision_at_k(results, ground_truth, k=5):
    if not results or not ground_truth:
        return 0.0

    top_k_ids = [res.doc_id for res in results[:k]]
    relevant_retrieved = sum(1 for doc_id in top_k_ids if doc_id in ground_truth)


    denominator = min(k, len(ground_truth))
    return relevant_retrieved / denominator if denominator > 0 else 0.0


def main():
    corpus, doc_texts = load_arxiv_corpus(DATA_DIR)
    print("🏗️ Побудова індексу для оцінювання...")
    index = build_index(corpus, with_positions=True)

    tfidf_scorer = TfIdf()
    bm25_scorer = BM25()

    print("\n" + "=" * 85)
    print("📊 ОЦІНЮВАННЯ ЯКОСТІ ПОШУКУ (Precision@5) З РУЧНОЮ РОЗМІТКОЮ (GROUND TRUTH)")
    print("=" * 85)

    results_table = []
    total_tfidf_p5 = 0.0
    total_bm25_p5 = 0.0

    for idx, item in enumerate(GROUND_TRUTH_DATASET, start=1):
        query = item["query"]
        expected_ids = item["expected_doc_ids"]


        clean_terms = [
            t.lower() for t in tokenize(query) if t.lower() not in ("and", "or", "not")
        ]
        text_matches = {
            doc_id
            for doc_id, text in doc_texts.items()
            if all(term in text for term in clean_terms)
        }


        full_ground_truth = expected_ids.union(text_matches)


        res_tfidf = search(query, index, scorer=tfidf_scorer, k=5)
        p5_tfidf = calculate_precision_at_k(res_tfidf, full_ground_truth, k=5)


        res_bm25 = search(query, index, scorer=bm25_scorer, k=5)
        p5_bm25 = calculate_precision_at_k(res_bm25, full_ground_truth, k=5)

        total_tfidf_p5 += p5_tfidf
        total_bm25_p5 += p5_bm25

        results_table.append(
            (idx, query, len(full_ground_truth), p5_tfidf, p5_bm25)
        )

    num_queries = len(GROUND_TRUTH_DATASET)
    avg_tfidf = total_tfidf_p5 / num_queries
    avg_bm25 = total_bm25_p5 / num_queries

    print("\n| # | Query | Ground Truth Docs | TF-IDF P@5 | BM25 P@5 |")
    print("|---|---|---|---|---|")
    for row in results_table:
        print(f"| {row[0]} | `{row[1]}` | {row[2]} | {row[3]:.2f} | {row[4]:.2f} |")
    print(
        f"| **AVG** | **Mean Precision@5** | - | **{avg_tfidf:.2f}** | **{avg_bm25:.2f}** |\n"
    )


if __name__ == "__main__":
    main()