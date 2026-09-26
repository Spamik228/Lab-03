import pytest
from findex3.index import build_index
from findex3.scoring import BM25, TfIdf, Scorer
from findex3.search import SearchResult, search


@pytest.fixture
def indexed_corpus():
    corpus = [
        {
            "doc_id": 1,
            "title": "Python Guide",
            "path": "/1.txt",
            "tokens": ["python", "programming", "language", "python"],
        },
        {
            "doc_id": 2,
            "title": "Java Overview",
            "path": "/2.txt",
            "tokens": ["java", "programming", "language"],
        },
        {
            "doc_id": 3,
            "title": "Snake Species",
            "path": "/3.txt",
            "tokens": ["python", "snake", "reptile"],
        },
    ]
    return build_index(corpus)


def test_scorers_implement_protocol():
    tfidf = TfIdf()
    bm25 = BM25()

    assert isinstance(tfidf, Scorer)
    assert isinstance(bm25, Scorer)


def test_search_bm25(indexed_corpus):
    results = search("python", indexed_corpus, scorer=BM25(), k=2)

    assert len(results) == 2
    assert results[0].doc_id == 1
    assert results[1].doc_id == 3
    assert results[0].score > results[1].score


def test_search_tfidf(indexed_corpus):

    results = search("java OR programming", indexed_corpus, scorer=TfIdf(), k=5)

    assert len(results) >= 2

    assert results[0].doc_id == 2

def test_search_result_sorting():
    res1 = SearchResult(score=1.5, doc_id=1, title="Doc 1")
    res2 = SearchResult(score=3.2, doc_id=2, title="Doc 2")


    sorted_res = sorted([res1, res2], reverse=True)
    assert sorted_res[0] == res2
    assert sorted_res[1] == res1