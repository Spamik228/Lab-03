import pytest
from pathlib import Path
from findex3.index import Index, Posting, DocMeta, build_index
from findex3.store import open_index

@pytest.fixture
def sample_corpus():
    return [
        {
            "doc_id": 1,
            "title": "Doc 1",
            "path": "/path/1.txt",
            "tokens": ["cat", "dog", "cat"],
        },
        {
            "doc_id": 2,
            "title": "Doc 2",
            "path": "/path/2.txt",
            "tokens": ["dog", "bird"],
        },
    ]


def test_index_dunder_methods(sample_corpus):
    index = build_index(sample_corpus)


    assert len(index) == 3


    assert "cat" in index
    assert "fish" not in index


    postings = index["cat"]
    assert isinstance(postings, list)
    assert postings[0].doc_id == 1
    assert postings[0].tf == 2

    with pytest.raises(KeyError):
        _ = index["non_existent_term"]


    terms = set(iter(index))
    assert terms == {"cat", "dog", "bird"}


    repr_str = repr(index)
    assert "Index(" in repr_str
    assert "terms=3" in repr_str


def test_index_properties_and_methods(sample_corpus):
    index = build_index(sample_corpus)


    assert index.num_docs == 2


    assert index.avg_doc_length == 2.5


    assert index.doc_length(1) == 3
    assert index.doc_length(2) == 2
    assert index.doc_length(99) == 0

    assert index.df("dog") == 2
    assert index.df("fish") == 0


def test_open_index_context_manager(tmp_path: Path, sample_corpus):
    index_file = tmp_path / "test_index.pkl"


    with open_index(index_file) as idx:
        temp_idx = build_index(sample_corpus)
        idx._postings.update(temp_idx._postings)
        idx._doc_lengths.update(temp_idx._doc_lengths)
        idx._doc_meta.update(temp_idx._doc_meta)


    assert index_file.exists()
    with open_index(index_file) as loaded_idx:
        assert loaded_idx.num_docs == 2
        assert "cat" in loaded_idx
        assert loaded_idx.df("dog") == 2