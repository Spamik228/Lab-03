import pytest
from findex3.index import build_index
from findex3.query import And, Not, Or, Phrase, Term, parse
from findex3.search import search


def test_parser_tree_equality():
    parsed = parse("a OR b c")
    expected = Or(Term("a"), And(Term("b"), Term("c")))
    assert parsed == expected


def test_parser_operators_and_not():
    parsed = parse("python NOT java")
    expected = And(Term("python"), Not(Term("java")))
    assert parsed == expected


def test_parser_parentheses_and_phrases():
    parsed = parse('(a OR b) AND "hello world"')
    expected = And(
        Or(Term("a"), Term("b")),
        Phrase(("hello", "world")),
    )
    assert parsed == expected


def test_operator_overloading():
    a = Term("a")
    b = Term("b")
    c = Term("c")

    tree = (a | b) & ~c
    expected = And(Or(a, b), Not(c))
    assert tree == expected


def test_phrase_query_evaluation():
    corpus = [
        {
            "doc_id": 1,
            "title": "Doc 1",
            "path": "/1.txt",
            "tokens": ["quick", "brown", "fox"],
        },
        {
            "doc_id": 2,
            "title": "Doc 2",
            "path": "/2.txt",
            "tokens": ["brown", "quick", "fox"],
        },
    ]
    index = build_index(corpus, with_positions=True)
    phrase_ast = parse('"quick brown"')
    matched_docs = phrase_ast.evaluate(index)

    assert matched_docs == {1}


def test_full_search_with_boolean_query():
    corpus = [
        {
            "doc_id": 1,
            "title": "Doc 1",
            "path": "/1.txt",
            "tokens": ["python", "search", "engine"],
        },
        {
            "doc_id": 2,
            "title": "Doc 2",
            "path": "/2.txt",
            "tokens": ["python", "django", "web"],
        },
        {
            "doc_id": 3,
            "title": "Doc 3",
            "path": "/3.txt",
            "tokens": ["java", "spring", "web"],
        },
    ]
    index = build_index(corpus, with_positions=True)

    results = search("python NOT web", index)
    assert len(results) == 1
    assert results[0].doc_id == 1