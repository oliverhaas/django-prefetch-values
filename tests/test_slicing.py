"""Tests for OFFSET/LIMIT slicing support with values_nested().

These tests verify that .values_nested()[start:end] applies OFFSET/LIMIT
directly to the SQL query, matching Django's native .values()[start:end] behavior.
"""

from __future__ import annotations

from django.db import connection, reset_queries

from django_nested_values import NestedValuesQuerySet
from tests.testapp.models import Book


def count_queries(func):
    """Count the number of queries executed by a function."""
    reset_queries()
    func()
    return len(connection.queries)


class TestValuesNestedSlicing:
    """Tests for slicing behavior with values_nested()."""

    def test_values_nested_slicing_returns_correct_results(self, sample_data):
        """Slicing values_nested() should return correct items at correct positions."""
        # Get all results to establish expected order
        qs = NestedValuesQuerySet(model=Book)
        all_results = list(qs.order_by("id").values_nested())
        assert len(all_results) == 3, "Expected 3 books in sample data"

        # Slice [1:3] should return items at index 1 and 2
        sliced_results = list(qs.order_by("id").values_nested()[1:3])

        assert len(sliced_results) == 2, "Expected 2 results from slice [1:3]"
        assert sliced_results[0]["id"] == all_results[1]["id"]
        assert sliced_results[1]["id"] == all_results[2]["id"]
        assert sliced_results[0]["title"] == all_results[1]["title"]
        assert sliced_results[1]["title"] == all_results[2]["title"]

    def test_values_nested_slicing_with_relations(self, sample_data):
        """Slicing should work correctly with prefetch_related."""
        qs = NestedValuesQuerySet(model=Book)

        # Get first book with relations
        sliced_results = list(
            qs.order_by("id").prefetch_related("authors", "chapters").values_nested()[0:1],
        )

        assert len(sliced_results) == 1
        book = sliced_results[0]
        assert "authors" in book
        assert "chapters" in book
        # The first book is "Django for Beginners" with 2 authors and 3 chapters
        assert isinstance(book["authors"], list)
        assert isinstance(book["chapters"], list)

    def test_values_nested_slicing_executes_optimal_query_count(self, sample_data, settings):
        """Slicing should produce same query count as Django's values() slicing."""
        settings.DEBUG = True

        # Django native values() with slicing
        def django_native():
            qs = Book.objects.order_by("id").values()[0:2]
            list(qs)

        native_count = count_queries(django_native)

        # Our implementation with slicing (no prefetch for fair comparison)
        def values_nested_query():
            qs = NestedValuesQuerySet(model=Book)
            list(qs.order_by("id").values_nested()[0:2])

        our_count = count_queries(values_nested_query)

        assert our_count == native_count, (
            f"Expected {native_count} queries (Django native values() slicing), got {our_count}"
        )

    def test_values_nested_slicing_with_prefetch_matches_django(self, sample_data, settings):
        """Slicing with prefetch should match Django's query behavior."""
        settings.DEBUG = True

        # Django native with prefetch and slicing
        def django_native():
            # When using prefetch_related, Django needs:
            # 1. Main query with LIMIT/OFFSET
            # 2. Prefetch queries for related objects
            qs = Book.objects.order_by("id").prefetch_related("authors")[:2]
            for book in qs:
                _ = [a.name for a in book.authors.all()]

        native_count = count_queries(django_native)

        # Our implementation
        def values_nested_query():
            qs = NestedValuesQuerySet(model=Book)
            list(qs.order_by("id").prefetch_related("authors").values_nested()[0:2])

        our_count = count_queries(values_nested_query)

        assert our_count == native_count, (
            f"Expected {native_count} queries (Django native with prefetch), got {our_count}"
        )

    def test_values_nested_slicing_applies_limit_offset_to_sql(self, sample_data, settings):
        """Slicing should apply LIMIT/OFFSET directly to SQL, not post-filter in Python."""
        settings.DEBUG = True

        qs = NestedValuesQuerySet(model=Book)
        nested_qs = qs.order_by("id").values_nested()

        # Apply slice
        sliced_qs = nested_qs[1:3]

        # Execute and capture query
        reset_queries()
        list(sliced_qs)

        # Check that LIMIT and OFFSET are in the SQL
        assert len(connection.queries) >= 1, "Expected at least one query"
        main_query = connection.queries[0]["sql"].upper()

        assert "LIMIT" in main_query, f"Expected LIMIT in SQL query, got: {connection.queries[0]['sql']}"
        # OFFSET 1 should be present for [1:3] slice
        # Note: Some databases might represent this differently
        assert "OFFSET" in main_query or "LIMIT 2 OFFSET 1" in main_query.replace(" ", ""), (
            f"Expected OFFSET in SQL query, got: {connection.queries[0]['sql']}"
        )

    def test_values_nested_slicing_start_only(self, sample_data):
        """Slicing with start index only [n:] should work correctly."""
        qs = NestedValuesQuerySet(model=Book)
        all_results = list(qs.order_by("id").values_nested())

        # Skip first item
        sliced_results = list(qs.order_by("id").values_nested()[1:])

        assert len(sliced_results) == len(all_results) - 1
        assert sliced_results[0]["id"] == all_results[1]["id"]

    def test_values_nested_slicing_end_only(self, sample_data):
        """Slicing with end index only [:n] should work correctly."""
        qs = NestedValuesQuerySet(model=Book)
        all_results = list(qs.order_by("id").values_nested())

        # Take first 2 items
        sliced_results = list(qs.order_by("id").values_nested()[:2])

        assert len(sliced_results) == 2
        assert sliced_results[0]["id"] == all_results[0]["id"]
        assert sliced_results[1]["id"] == all_results[1]["id"]

    def test_values_nested_slicing_single_item(self, sample_data):
        """Indexing with single integer [n] should work correctly."""
        qs = NestedValuesQuerySet(model=Book)
        all_results = list(qs.order_by("id").values_nested())

        # Get second item
        single_result = qs.order_by("id").values_nested()[1]

        assert isinstance(single_result, dict)
        assert single_result["id"] == all_results[1]["id"]
        assert single_result["title"] == all_results[1]["title"]

    def test_values_nested_slicing_empty_result(self, sample_data):
        """Slicing beyond available results should return empty list."""
        qs = NestedValuesQuerySet(model=Book)

        # Slice beyond available data
        sliced_results = list(qs.order_by("id").values_nested()[100:200])

        assert sliced_results == []

    def test_values_nested_slicing_preserves_filters(self, sample_data):
        """Slicing should work with filters applied."""
        qs = NestedValuesQuerySet(model=Book)

        # Filter and slice
        filtered_sliced = list(
            qs.filter(publisher__name="Tech Books Inc").order_by("id").values_nested()[0:1],
        )

        assert len(filtered_sliced) == 1
        # Both Django for Beginners and Advanced Python are from Tech Books Inc
        # Should get the first one after ordering by id

    def test_values_nested_chained_operations_order(self, sample_data, settings):
        """Test that operations can be chained in any order."""
        settings.DEBUG = True

        qs = NestedValuesQuerySet(model=Book)

        # These should all produce equivalent results
        result1 = list(qs.order_by("id").values_nested()[:2])
        result2 = list(qs.values_nested().order_by("id")[:2])

        # Both should return 2 results
        assert len(result1) == 2
        assert len(result2) == 2

        # Results should have same IDs (order matters)
        assert result1[0]["id"] == result2[0]["id"]
        assert result1[1]["id"] == result2[1]["id"]

    def test_values_nested_slicing_with_select_related(self, sample_data, settings):
        """Slicing should preserve select_related and return nested dicts."""
        settings.DEBUG = True
        reset_queries()

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.order_by("id").select_related("publisher").values_nested()[0:2])

        # Should use single query with JOIN
        assert len(connection.queries) == 1
        sql = connection.queries[0]["sql"]
        assert "JOIN" in sql.upper(), f"Expected JOIN in SQL: {sql}"
        assert "LIMIT" in sql.upper(), f"Expected LIMIT in SQL: {sql}"

        # Check results have nested publisher dict
        assert len(result) == 2
        for book in result:
            assert "publisher" in book, f"Missing 'publisher' key in {book}"
            assert isinstance(book["publisher"], dict), f"Publisher should be dict, got {type(book.get('publisher'))}"
            assert "name" in book["publisher"], f"Publisher missing 'name': {book['publisher']}"
            assert "country" in book["publisher"], f"Publisher missing 'country': {book['publisher']}"

    def test_values_nested_slicing_select_related_matches_unsliced(self, sample_data):
        """Sliced results with select_related should match unsliced results."""
        qs = NestedValuesQuerySet(model=Book)

        # Get all results
        all_results = list(qs.order_by("id").select_related("publisher").values_nested())

        # Get sliced results
        sliced_results = list(qs.order_by("id").select_related("publisher").values_nested()[0:2])

        # First two should be identical
        assert sliced_results[0] == all_results[0], "First item mismatch"
        assert sliced_results[1] == all_results[1], "Second item mismatch"

        # Both should have publisher as dict with same content
        assert sliced_results[0]["publisher"] == all_results[0]["publisher"]
        assert sliced_results[1]["publisher"] == all_results[1]["publisher"]
