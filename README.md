
### 🧪 Precision@5 table (10 hand-labeled queries) for TF-IDF vs. BM25.

INFO:findex3:[build_index] виконалася за 107.78 ms

| # | Query                          | Ground Truth Docs | TF-IDF P@5 | BM25 P@5 |
|---|--------------------------------|-------------------|------------|----------|
| 1 | `data analysis`                | 37                | 1.00       | 1.00     |
| 2 | `neural network`               | 3                 | 1.00       | 1.00     |
| 3 | `quantum system`               | 33                | 1.00       | 1.00     |
| 4 | `deep learning model`          | 0                 | 0.00       | 0.00     |
| 5 | `algorithm performance`        | 4                 | 1.00       | 1.00     |
| 6 | `image classification`         | 0                 | 0.00       | 0.00     |
| 7 | `time series analysis`         | 1                 | 0.00       | 0.00     |
| 8 | `optimization problem`         | 3                 | 1.00       | 1.00     |
| 9 | `graph neural network`         | 2                 | 1.00       | 1.00     |
| 10 | `natural language processing` | 0                 | 0.00       | 0.00     |
| **AVG** | **Mean Precision@5**     | -                 | **0.60**   | **0.60** |

| # | Query                       | Ground Truth Docs | TF-IDF P@5 | BM25 P@5 |
|---|-----------------------------|-------------------|------------|----------|
| 1 | `quantum AND computing`     | 5                 | 0.20       | 0.20     |
| 2 | `neural AND networks`       | 5                 | 0.20       | 0.20     |
| 3 | `learning OR training`      | 6                 | 0.20       | 0.20     |
| 4 | `model AND data`            | 63                | 1.00       | 1.00     |
| 5 | `algorithm NOT java`        | 5                 | 0.00       | 0.00     |
| 6 | `graph AND node`            | 5                 | 0.00       | 0.00     |
| 7 | `optimization`              | 13                | 1.00       | 1.00     |
| 8 | `classification AND method` | 5                 | 0.00       | 0.00     |
| 9 | `analysis AND system`       | 31                | 1.00       | 1.00     |
| 10 | `image OR signal`          | 6                 | 0.40       | 0.40     |
| **AVG** | **Mean Precision@5**  | -                 | **0.40**   | **0.40** |


### 🧪 Ranking Sanity Checks Log

```text
================================================================================
🧪 RANKING SANITY CHECKS (arXiv Dataset)
================================================================================

1️⃣  IDF Effect (Рідкісний vs Частотний термін)
   ├── 'qubit' (DF=8)    │ BM25 Score: 5.6776
   └── 'paper' (DF=144)  │ BM25 Score: 2.7555
   ✔ STATUS: PASSED (Рідкісний термін має значно вищий скор за рахунок IDF)

2️⃣  TF Saturation (Насичення частоти слова 'model' у Doc #648)
   ├── TF = 1            │ BM25 Score: 1.0371
   ├── TF = 2            │ BM25 Score: 1.6246  (+0.5876)
   └── TF = 8            │ BM25 Score: 2.8251  (+0.2001 / entry)
   ✔ STATUS: PASSED (Сублінійне зростання скору при збільшенні TF)

3️⃣  Length Penalty (Штраф за довжину документа для 'algorithm')
   ├── Doc #379 (7 words, TF=1)   │ BM25 Score: 5.8467
   └── Doc #954 (265 words, TF=1) │ BM25 Score: 2.1940
   ✔ STATUS: PASSED (Коротший документ отримав вищий скор, b=0.75)
================================================================================
