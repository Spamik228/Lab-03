import logging
import sys
from pathlib import Path

from findex3.index import build_index
from findex3.scoring import BM25, TfIdf
from findex3.search import _parse_cached, search
from findex3.store import load, save
from findex3.tokenizer import tokenize

DATA_DIR = Path(r"D:\pr\findex1\data")
INDEX_PATH = Path("arxiv_10k.idx")


def load_arxiv_corpus(data_dir: Path, max_docs: int = 10000):

    print(f"📂 Зчитування файлів з {data_dir}...")
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
        doc_texts[doc_id] = content

    print(f"✅ Успішно зчитано {len(corpus)} документів.")
    return corpus, doc_texts


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if INDEX_PATH.exists():
        print(f"⚡ Знайдено збережений індекс {INDEX_PATH}, завантажуємо...")
        index = load(INDEX_PATH)

        _, doc_texts = load_arxiv_corpus(DATA_DIR)
    else:
        corpus, doc_texts = load_arxiv_corpus(DATA_DIR)
        print("🏗️ Будуємо інвертований індекс (з позиціями)...")
        index = build_index(corpus, with_positions=True)
        print(f"💾 Зберігаємо індекс у {INDEX_PATH}...")
        save(index, INDEX_PATH)

    print("\n" + "=" * 60)
    print(f"📊 Статистика індексу:")
    print(f"   • Документів: {index.num_docs}")
    print(f"   • Унікальних термінів: {len(index)}")
    print(f"   • Середня довжина документа: {index.avg_doc_length:.2f} слів")
    print("=" * 60 + "\n")


    print("🔍 Інтерактивний пошук за arXiv abstracts!")
    print("Підтримуються булеві запити (AND, OR, NOT, дужки) та фрази у \"лапках\".")
    print("Введіть 'exit' або 'quit' для виходу.\n")

    while True:
        try:
            query_str = input("\n[findex3] Запит > ").strip()
            if not query_str or query_str.lower() in ("exit", "quit"):
                break


            results = search(
                query=query_str,
                index=index,
                scorer=BM25(),
                k=5,
                include_snippets=True,
                doc_texts=doc_texts,
            )


            _ = search(
                query=query_str,
                index=index,
                scorer=BM25(),
                k=5,
                include_snippets=False,
            )

            if not results:
                print("❌ Нічого не знайдено.")
                print(f"⚡ Стан LRU-кешу: {_parse_cached.cache_info()}")
                continue

            print(f"\nЗнайдено {len(results)} результатів (Top-5):")
            print("-" * 60)
            for i, res in enumerate(results, 1):
                print(f"{i}. [{res.score:.4f}] Doc #{res.doc_id}: {res.title}")
                if res.snippet:
                    print(f"   Сніпет: {res.snippet}")
                print("-" * 60)


            print(f"\n⚡ [LRU Cache Statistics]: {_parse_cached.cache_info()}")

        except Exception as e:
            print(f"⚠️ Помилка виконання запиту: {e}")


if __name__ == "__main__":
    main()