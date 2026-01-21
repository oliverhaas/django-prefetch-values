"""Tests for attribute-style access via values_nested(as_attr_dicts=True).

Tests cover:
- AttrDict as a dict subclass
- Attribute access on AttrDict
- Dict access on AttrDict
- Nested object/list access
- Compatibility with existing dict patterns
"""

from __future__ import annotations

import pytest

from django_nested_values import AttrDict, NestedValuesQuerySet
from tests.testapp.models import Author, Book


class TestAttrDict:
    """Test AttrDict dict subclass."""

    def test_is_dict_subclass(self):
        """AttrDict is a dict subclass."""
        obj = AttrDict({"a": 1})
        assert isinstance(obj, dict)

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

    def test_nested_dict_not_auto_wrapped(self):
        """Nested dicts are NOT auto-wrapped (it's a simple dict subclass)."""
        obj = AttrDict({"publisher": {"name": "Tech Books", "country": "USA"}})
        # Nested dict is a plain dict, not AttrDict
        assert isinstance(obj.publisher, dict)
        assert not isinstance(obj.publisher, AttrDict)

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
        """AttrDict equals plain dict with same content."""
        obj = AttrDict({"a": 1, "b": 2})
        assert obj == {"a": 1, "b": 2}

    def test_setattr(self):
        """Setting attributes works."""
        obj = AttrDict({"a": 1})
        obj.b = 2
        assert obj.b == 2
        assert obj["b"] == 2  # Also accessible via dict

    def test_setitem(self):
        """Setting items works."""
        obj = AttrDict({"a": 1})
        obj["b"] = 2
        assert obj["b"] == 2
        assert obj.b == 2  # Also accessible via attribute

    def test_delattr(self):
        """Deleting attributes works."""
        obj = AttrDict({"a": 1, "b": 2})
        del obj.a
        assert "a" not in obj
        with pytest.raises(AttributeError):
            del obj.nonexistent

    def test_delitem(self):
        """Deleting items works."""
        obj = AttrDict({"a": 1, "b": 2})
        del obj["a"]
        assert "a" not in obj

    def test_repr(self):
        """repr() returns readable representation."""
        obj = AttrDict({"a": 1})
        # Should look like a dict repr
        assert "a" in repr(obj)
        assert "1" in repr(obj)

    def test_json_serializable(self):
        """AttrDict can be serialized to JSON."""
        import json

        obj = AttrDict({"a": 1, "b": "test"})
        result = json.dumps(obj)
        assert result == '{"a": 1, "b": "test"}'


class TestValuesNestedAsAttrDicts:
    """Test values_nested(as_attr_dicts=True) integration."""

    def test_basic_as_attr_dicts(self, sample_data):
        """values_nested(as_attr_dicts=True) returns AttrDict instances."""
        qs = NestedValuesQuerySet(model=Book)
        results = list(qs.values_nested(as_attr_dicts=True))

        assert len(results) > 0
        assert all(isinstance(r, AttrDict) for r in results)
        # Also all dicts (since AttrDict is a dict subclass)
        assert all(isinstance(r, dict) for r in results)

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
        """prefetch_related returns list of AttrDicts."""
        qs = NestedValuesQuerySet(model=Book)
        results = list(qs.prefetch_related("authors").values_nested(as_attr_dicts=True))
        book = results[0]

        assert isinstance(book.authors, list)
        assert all(isinstance(a, AttrDict) for a in book.authors)

    def test_prefetch_related_iteration(self, sample_data):
        """prefetch_related lists can be iterated directly."""
        qs = NestedValuesQuerySet(model=Book)
        results = list(qs.prefetch_related("authors").values_nested(as_attr_dicts=True))
        book = results[0]

        # Iterate directly (no .all() needed)
        for author in book.authors:
            assert isinstance(author, AttrDict)
            assert hasattr(author, "name")

    def test_nested_prefetch_objects(self, sample_data):
        """Deeply nested prefetch returns proper objects."""
        qs = NestedValuesQuerySet(model=Author)
        results = list(qs.prefetch_related("books__publisher").values_nested(as_attr_dicts=True))
        author = results[0]

        assert isinstance(author.books, list)
        for book in author.books:
            assert isinstance(book, AttrDict)
            assert isinstance(book.publisher, AttrDict)
            assert hasattr(book.publisher, "name")

    def test_django_pattern_compatibility(self, sample_data):
        """Iteration works similarly to Django ORM (without .all())."""
        qs = NestedValuesQuerySet(model=Author)
        results = list(qs.prefetch_related("books__chapters").values_nested(as_attr_dicts=True))

        # This pattern should work
        for author in results:
            for book in author.books:
                for chapter in book.chapters:
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

    def test_attr_dict_equals_plain_dict(self, sample_data):
        """AttrDict results equal plain dict results."""
        qs = NestedValuesQuerySet(model=Book)
        results_attr = list(qs.prefetch_related("authors").values_nested(as_attr_dicts=True))
        results_dict = list(qs.prefetch_related("authors").values_nested())

        # Results should be equal (AttrDict == dict with same content)
        for attr_obj, dict_obj in zip(results_attr, results_dict, strict=True):
            assert attr_obj == dict_obj
