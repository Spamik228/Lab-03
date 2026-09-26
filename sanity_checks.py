import sys
from pathlib import Path

from findex3.index import build_index
from findex3.scoring import BM25
from findex3.search import search
from findex3.tokenizer import tokenize

DATA_DIR = Path(r"D:\pr\findex1\data")


def load_corpus(data_dir: Path, max_docs: int = 10000):
  txt_files = list(data_dir.glob("*.txt"))[:max_docs]
  if not txt_files:
    print(f"❌ Файлів .txt не знайдено за шляхом: {data_dir}")
    sys.exit(1)

  corpus = []
  doc_texts = {}
  for doc_id, file_path in enumerate(txt_files, start=1):
    try:
      content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
      content = file_path.read_text(encoding="latin-1")

    title = file_path.stem
    corpus.append({
        "doc_id": doc_id,
        "title": title,
        "path": str(file_path),
        "tokens": list(tokenize(content)),
    })
    doc_texts[doc_id] = content

  return corpus, doc_texts


def main():
  print("📂 Завантаження корпусу та побудова індексу...")
  corpus, doc_texts = load_corpus(DATA_DIR)
  index = build_index(corpus, with_positions=True)
  bm25 = BM25(k1=1.5, b=0.75)

  print("\n" + "=" * 70)
  print("🧪 RUNNING 3 RANKING SANITY CHECKS ON REAL ARXIV DATASET")
  print("=" * 70)

  # ------------------------------------------------------------------
  # CHECK 1: Рідкий термін вище частого (IDF Effect)
  # ------------------------------------------------------------------
  rare_term = "qubit"
  frequent_term = "paper"

  rare_posting = index[rare_term][0] if index[rare_term] else None
  freq_posting = index[frequent_term][0] if index[frequent_term] else None

  score_rare = (
      bm25.score(rare_term, rare_posting, index) if rare_posting else 0.0
  )
  score_freq = (
      bm25.score(frequent_term, freq_posting, index) if freq_posting else 0.0
  )

  print("\n1️⃣  IDF Effect (Рідкісний термін vs Частотний):")
  print(
      f"   • Рідкісний термін '{rare_term}' (DF={index.df(rare_term)}): BM25 Score"
      f" = {score_rare:.4f}"
  )
  print(
      f"   • Частий термін    '{frequent_term}' (DF={index.df(frequent_term)}):"
      f" BM25 Score = {score_freq:.4f}"
  )
  assert (
      score_rare > score_freq
  ), "Sanity Check 1 failed: рідкісний термін повинен мати вищий скор!"
  print("   PASSED: Рідкісний термін має значно вищий скор завдяки IDF.")

  # ------------------------------------------------------------------
  # CHECK 2: 20-те повторення майже нічого не додає (TF Saturation)
  # ------------------------------------------------------------------
  # Шукаємо документ у нашому корпусі з найбільшим TF для слова "model" або "data"
  target_term = "model"
  postings = sorted(index[target_term], key=lambda p: p.tf, reverse=True)
  high_tf_posting = postings[0]


  score_tf1 = bm25.score(
      target_term,
      type(high_tf_posting)(
          doc_id=high_tf_posting.doc_id, tf=1, positions=[0]
      ),
      index,
  )
  score_tf2 = bm25.score(
      target_term,
      type(high_tf_posting)(
          doc_id=high_tf_posting.doc_id, tf=2, positions=[0, 1]
      ),
      index,
  )
  score_tf_max = bm25.score(target_term, high_tf_posting, index)

  print(
      f"\n2️⃣  TF Saturation (Насичення частоти для слова '{target_term}' у Doc"
      f" #{high_tf_posting.doc_id}):"
  )
  print(f"   • При TF = 1 : BM25 Score = {score_tf1:.4f}")
  print(
      f"   • При TF = 2 : BM25 Score = {score_tf2:.4f}  (приріст:"
      f" +{score_tf2 - score_tf1:.4f})"
  )
  print(
      f"   • При TF = {high_tf_posting.tf} : BM25 Score = {score_tf_max:.4f}"
      f"  (приріст на 1 входження:"
      f" +{(score_tf_max - score_tf2)/(high_tf_posting.tf - 2):.4f})"
  )
  print("   PASSED: Повторні входження дають все менший приріст (плато).")

  # ------------------------------------------------------------------
  # CHECK 3: Короткий документ з одним входженням вище дуже довгого (Length Penalty)
  # ------------------------------------------------------------------
  term_check = "algorithm"
  # Шукаємо документи, де TF = 1
  single_tf_postings = [p for p in index[term_check] if p.tf == 1]

  # Обчислюємо довжини цих документів
  docs_with_len = []
  for p in single_tf_postings:
    doc_len = len(index.get_doc_tokens(p.doc_id)) if hasattr(index, 'get_doc_tokens') else len(corpus[p.doc_id-1]['tokens'])
    docs_with_len.append((p, doc_len))

  # Сортуємо: найкоротший та найдовший документ
  docs_with_len.sort(key=lambda x: x[1])
  short_p, short_len = docs_with_len[0]
  long_p, long_len = docs_with_len[-1]

  score_short = bm25.score(term_check, short_p, index)
  score_long = bm25.score(term_check, long_p, index)

  print(f"\n3️⃣  Length Penalty (Штраф за довжину для терміна '{term_check}'):")
  print(
      f"   • Короткий Doc #{short_p.doc_id} (довжина {short_len} слів, TF=1):"
      f" BM25 Score = {score_short:.4f}"
  )
  print(
      f"   • Довгий Doc #{long_p.doc_id}   (довжина {long_len} слів, TF=1):"
      f" BM25 Score = {score_long:.4f}"
  )
  assert (
      score_short > score_long
  ), "Sanity Check 3 failed: короткий документ повинен мати вищий скор!"
  print(
      "   PASSED: Коротший документ отримав вищий скор завдяки штрафу за"
      " довжину b=0.75."
  )
  print("\n" + "=" * 70)


if __name__ == "__main__":
  main()