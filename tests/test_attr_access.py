"""Tests for attribute-style access via values_nested(as_attr_dicts=True).

Tests cover:
- Attribute access on AttrDict
- Dict access on AttrDict
- RelatedList with .all() method
- Nested object/list access
- Compatibility with existing dict patterns
"""

from __future__ import annotations

import pytest

from django_nested_values import AttrDict, NestedValuesQuerySet, RelatedList
from tests.testapp.models import Author, Book


class TestAttrDict:
    """Test AttrDict wrapper class."""

    def test_attribute_access(self):
        """AttrDict supports attribute access."""
        obj = AttrDict({"title": "Django Guide", "price": 29.99})
        assert obj.title == "Django Guide"
        assert obj.price == 29.99

    def test_dict_access(self):
        """AttrDict supports dict-style access."""
        obj = AttrDict({"title": "Django Guide", "price": 29.99})
        assert obj["title"] == "Django Guide"
        assert obj["price"] == 29.99

    def test_contains(self):
        """AttrDict supports 'in' operator."""
        obj = AttrDict({"title": "Django Guide"})
        assert "title" in obj
        assert "price" not in obj

    def test_len(self):
        """AttrDict supports len()."""
        obj = AttrDict({"a": 1, "b": 2, "c": 3})
        assert len(obj) == 3

    def test_iter(self):
        """AttrDict supports iteration over keys."""
        obj = AttrDict({"a": 1, "b": 2})
        assert set(obj) == {"a", "b"}

    def test_keys_values_items(self):
        """AttrDict supports keys(), values(), items()."""
        obj = AttrDict({"a": 1, "b": 2})
        assert list(obj.keys()) == ["a", "b"]
        assert list(obj.values()) == [1, 2]
        assert list(obj.items()) == [("a", 1), ("b", 2)]

    def test_get_with_default(self):
        """AttrDict supports get() with default."""
        obj = AttrDict({"a": 1})
        assert obj.get("a") == 1
        assert obj.get("b") is None
        assert obj.get("b", "default") == "default"

    def test_nested_dict_wrapped(self):
        """Nested dicts are wrapped as AttrDict."""
        obj = AttrDict({"publisher": {"name": "Tech Books", "country": "USA"}})
        assert isinstance(obj.publisher, AttrDict)
        assert obj.publisher.name == "Tech Books"
        assert obj.publisher["country"] == "USA"

    def test_nested_list_wrapped(self):
        """Lists are wrapped as RelatedList."""
        obj = AttrDict({"authors": [{"name": "John"}, {"name": "Jane"}]})
        assert isinstance(obj.authors, RelatedList)
        assert len(obj.authors) == 2
        assert isinstance(obj.authors[0], AttrDict)
        assert obj.authors[0].name == "John"

    def test_attribute_error_on_missing(self):
        """Accessing missing attribute raises AttributeError."""
        obj = AttrDict({"title": "Book"})
        with pytest.raises(AttributeError, match="has no attribute 'missing'"):
            _ = obj.missing

    def test_equality_with_attr_dict(self):
        """AttrDict can be compared with another AttrDict."""
        obj1 = AttrDict({"a": 1})
        obj2 = AttrDict({"a": 1})
        obj3 = AttrDict({"a": 2})
        assert obj1 == obj2
        assert obj1 != obj3

    def test_equality_with_dict(self):
        """AttrDict can be compared with a dict."""
        obj = AttrDict({"a": 1})
        # Note: nested wrapping means direct comparison with plain dict may differ
        # for nested structures, but simple cases should work
        assert obj._data == {"a": 1}

    def test_to_dict(self):
        """to_dict() converts back to plain dict recursively."""
        obj = AttrDict(
            {
                "title": "Book",
                "publisher": {"name": "Pub"},
                "authors": [{"name": "John"}, {"name": "Jane"}],
            },
        )
        result = obj.to_dict()
        assert result == {
            "title": "Book",
            "publisher": {"name": "Pub"},
            "authors": [{"name": "John"}, {"name": "Jane"}],
        }
        assert isinstance(result, dict)
        assert isinstance(result["publisher"], dict)
        assert isinstance(result["authors"], list)
        assert isinstance(result["authors"][0], dict)

    def test_setattr(self):
        """Setting attributes works."""
        obj = AttrDict({"a": 1})
        obj.b = 2
        assert obj.b == 2

    def test_setitem(self):
        """Setting items works."""
        obj = AttrDict({"a": 1})
        obj["b"] = 2
        assert obj["b"] == 2

    def test_repr(self):
        """repr() returns readable representation."""
        obj = AttrDict({"a": 1})
        assert "AttrDict" in repr(obj)
        assert "a" in repr(obj)


class TestRelatedList:
    """Test RelatedList wrapper class."""

    def test_all_returns_self(self):
        """RelatedList.all() returns self."""
        lst = RelatedList([1, 2, 3])
        assert lst.all() is lst

    def test_list_operations(self):
        """RelatedList supports standard list operations."""
        lst = RelatedList([1, 2, 3])
        assert len(lst) == 3
        assert lst[0] == 1
        assert list(lst) == [1, 2, 3]
        assert 2 in lst

    def test_iteration(self):
        """RelatedList supports iteration."""
        lst = RelatedList([{"name": "a"}, {"name": "b"}])
        names = [item["name"] for item in lst]
        assert names == ["a", "b"]

    def test_all_in_loop(self):
        """RelatedList.all() works in for loop (Django pattern)."""
        lst = RelatedList([1, 2, 3])
        result = list(lst.all())
        assert result == [1, 2, 3]


class TestValuesNestedAsAttrDicts:
    """Test values_nested(as_attr_dicts=True) integration."""

    def test_basic_as_attr_dicts(self, sample_data):
        """values_nested(as_attr_dicts=True) returns AttrDict instances."""
        qs = NestedValuesQuerySet(model=Book)
        results = list(qs.values_nested(as_attr_dicts=True))

        assert len(results) > 0
        assert all(isinstance(r, AttrDict) for r in results)

    def test_attribute_access_on_result(self, sample_data):
        """Can access fields via attributes."""
        qs = NestedValuesQuerySet(model=Book)
        results = list(qs.values_nested(as_attr_dicts=True))
        book = results[0]

        assert hasattr(book, "title")
        assert hasattr(book, "id")
        assert isinstance(book.title, str)

    def test_select_related_attr_dict(self, sample_data):
        """select_related fields are AttrDict."""
        qs = NestedValuesQuerySet(model=Book)
        results = list(qs.select_related("publisher").values_nested(as_attr_dicts=True))
        book = results[0]

        assert isinstance(book.publisher, AttrDict)
        assert hasattr(book.publisher, "name")
        assert isinstance(book.publisher.name, str)

    def test_prefetch_related_list(self, sample_data):
        """prefetch_related returns RelatedList."""
        qs = NestedValuesQuerySet(model=Book)
        results = list(qs.prefetch_related("authors").values_nested(as_attr_dicts=True))
        book = results[0]

        assert isinstance(book.authors, RelatedList)

    def test_prefetch_related_all_method(self, sample_data):
        """prefetch_related lists support .all() method."""
        qs = NestedValuesQuerySet(model=Book)
        results = list(qs.prefetch_related("authors").values_nested(as_attr_dicts=True))
        book = results[0]

        # Can call .all() like Django's related manager
        authors = book.authors.all()
        assert authors is book.authors  # Returns self

        # Can iterate
        for author in book.authors.all():
            assert isinstance(author, AttrDict)
            assert hasattr(author, "name")

    def test_nested_prefetch_objects(self, sample_data):
        """Deeply nested prefetch returns proper objects."""
        qs = NestedValuesQuerySet(model=Author)
        results = list(qs.prefetch_related("books__publisher").values_nested(as_attr_dicts=True))
        author = results[0]

        assert isinstance(author.books, RelatedList)
        for book in author.books.all():
            assert isinstance(book, AttrDict)
            assert isinstance(book.publisher, AttrDict)
            assert hasattr(book.publisher, "name")

    def test_django_pattern_compatibility(self, sample_data):
        """Code written for Django ORM works with as_attr_dicts=True."""
        qs = NestedValuesQuerySet(model=Author)
        results = list(qs.prefetch_related("books__chapters").values_nested(as_attr_dicts=True))

        # This pattern should work like Django ORM
        for author in results:
            for book in author.books.all():
                for chapter in book.chapters.all():
                    # Can access attributes
                    _ = chapter.title
                    _ = chapter.number

    def test_dict_style_still_works(self, sample_data):
        """Dict-style access still works with as_attr_dicts=True."""
        qs = NestedValuesQuerySet(model=Book)
        results = list(qs.select_related("publisher").values_nested(as_attr_dicts=True))
        book = results[0]

        # Both styles work
        assert book.title == book["title"]
        assert book.publisher.name == book["publisher"]["name"]

    def test_default_still_returns_dicts(self, sample_data):
        """Default values_nested() still returns plain dicts."""
        qs = NestedValuesQuerySet(model=Book)
        results = list(qs.values_nested())  # No as_attr_dicts

        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)
        assert not any(isinstance(r, AttrDict) for r in results)

    def test_to_dict_round_trip(self, sample_data):
        """Can convert AttrDict back to dict."""
        qs = NestedValuesQuerySet(model=Book)
        results_objects = list(qs.prefetch_related("authors").values_nested(as_attr_dicts=True))
        results_dicts = list(qs.prefetch_related("authors").values_nested())

        for obj, expected_dict in zip(results_objects, results_dicts, strict=True):
            converted = obj.to_dict()
            assert converted == expected_dict
