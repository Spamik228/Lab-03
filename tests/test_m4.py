import logging
import pytest
from findex3.index import build_index
from findex3.search import search, _parse_cached
from findex3.snippet import make_snippet
from findex3.scoring import BM25


@pytest.fixture
def sample_data():
    corpus = [
        {
            "doc_id": 1,
            "title": "Python Overview",
            "path": "/1.txt",
            "tokens": ["python", "is", "a", "programming", "language"],
        },
        {
            "doc_id": 2,
            "title": "Java Guide",
            "path": "/2.txt",
            "tokens": ["java", "is", "another", "language"],
        },
    ]
    index = build_index(corpus, with_positions=True)
    doc_texts = {
        1: "Python is a powerful programming language used worldwide.",
        2: "Java is another widespread enterprise programming language.",
    }
    return index, doc_texts


def test_snippets_highlighting():
    text = "Python is a powerful programming language used worldwide."
    snippet = make_snippet(text, ["programming"], window=20)

    assert "**programming**" in snippet or "**Programming**" in snippet


def test_search_with_snippets(sample_data):
    index, doc_texts = sample_data
    results = search("programming", index, doc_texts=doc_texts, include_snippets=True)

    assert len(results) > 0
    assert results[0].snippet != ""
    assert "**" in results[0].snippet

def test_lru_cache_hit(sample_data):
    index, _ = sample_data
    _parse_cached.cache_clear()


    res1 = search("language", index)

    res2 = search("language", index)

    cache_info = _parse_cached.cache_info()
    assert cache_info.hits >= 1
    assert res1 == res2