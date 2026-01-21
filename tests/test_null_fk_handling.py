"""Tests for NULL ForeignKey handling in values_nested().

These tests verify that when a ForeignKey field is NULL in the database,
values_nested() includes it in the result dict as None, rather than omitting
the key entirely.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from django_nested_values import NestedValuesQuerySet
from tests.testapp.models import Author, Book, Publisher


@pytest.mark.django_db
class TestNullFKHandling:
    """Tests for NULL ForeignKey handling."""

    def test_null_fk_included_as_none_in_dict(self):
        """NULL FK fields should appear as None in the result dict, not be omitted."""
        publisher = Publisher.objects.create(name="Test Publisher", country="USA")
        book = Book.objects.create(
            title="Orphan Book",
            isbn="1111111111111",
            price=Decimal("19.99"),
            published_date=date(2024, 1, 1),
            publisher=publisher,
            editor=None,  # NULL FK
        )

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.filter(id=book.id).select_related("editor").values_nested())

        assert len(result) == 1
        assert "editor" in result[0], "NULL FK should be present in dict"
        assert result[0]["editor"] is None

    def test_null_fk_included_as_none_in_attrdict(self):
        """NULL FK fields should appear as None in AttrDict, not raise AttributeError."""
        publisher = Publisher.objects.create(name="Test Publisher", country="USA")
        book = Book.objects.create(
            title="Orphan Book",
            isbn="1111111111111",
            price=Decimal("19.99"),
            published_date=date(2024, 1, 1),
            publisher=publisher,
            editor=None,  # NULL FK
        )

        qs = NestedValuesQuerySet(model=Book)
        result = list(
            qs.filter(id=book.id).select_related("editor").values_nested(as_attr_dicts=True),
        )

        assert len(result) == 1
        # Should NOT raise AttributeError - editor should exist as None
        assert result[0].editor is None

    def test_non_null_fk_still_works(self):
        """Non-NULL FK fields should still be nested properly."""
        publisher = Publisher.objects.create(name="Test Publisher", country="USA")
        editor = Author.objects.create(name="Jane Editor", email="jane@example.com")
        book = Book.objects.create(
            title="Good Book",
            isbn="2222222222222",
            price=Decimal("29.99"),
            published_date=date(2024, 1, 1),
            publisher=publisher,
            editor=editor,  # Non-NULL FK
        )

        qs = NestedValuesQuerySet(model=Book)
        result = list(
            qs.filter(id=book.id).select_related("editor").values_nested(as_attr_dicts=True),
        )

        assert len(result) == 1
        assert result[0].editor.name == "Jane Editor"

    def test_mixed_null_and_non_null_fks(self):
        """Test a query returning both NULL and non-NULL FKs."""
        publisher = Publisher.objects.create(name="Test Publisher", country="USA")
        editor = Author.objects.create(name="Jane Editor", email="jane@example.com")

        book_with_editor = Book.objects.create(
            title="Book With Editor",
            isbn="3333333333333",
            price=Decimal("39.99"),
            published_date=date(2024, 1, 1),
            publisher=publisher,
            editor=editor,
        )
        book_without_editor = Book.objects.create(
            title="Book Without Editor",
            isbn="4444444444444",
            price=Decimal("19.99"),
            published_date=date(2024, 1, 1),
            publisher=publisher,
            editor=None,
        )

        qs = NestedValuesQuerySet(model=Book)
        results = list(
            qs.filter(id__in=[book_with_editor.id, book_without_editor.id])
            .select_related("editor")
            .order_by("title")
            .values_nested(),
        )

        assert len(results) == 2

        # Book With Editor should have nested editor dict
        assert results[0]["title"] == "Book With Editor"
        assert results[0]["editor"]["name"] == "Jane Editor"

        # Book Without Editor should have editor=None
        assert results[1]["title"] == "Book Without Editor"
        assert "editor" in results[1], "NULL FK should be present in dict"
        assert results[1]["editor"] is None

    def test_nested_null_fk_via_prefetch(self):
        """Test NULL FK when using prefetch_related instead of select_related."""
        publisher = Publisher.objects.create(name="Test Publisher", country="USA")
        book = Book.objects.create(
            title="Prefetched Book",
            isbn="5555555555555",
            price=Decimal("29.99"),
            published_date=date(2024, 1, 1),
            publisher=publisher,
            editor=None,  # NULL FK
        )

        qs = NestedValuesQuerySet(model=Book)
        result = list(
            qs.filter(id=book.id).prefetch_related("editor").values_nested(),
        )

        assert len(result) == 1
        assert "editor" in result[0], "NULL FK should be present in dict when prefetched"
        assert result[0]["editor"] is None
