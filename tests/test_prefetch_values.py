"""TDD tests for prefetch_related().values_nested() functionality.

These tests define the expected behavior of combining prefetch_related() with values_nested().
The implementation should make these tests pass.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from tests.testapp.models import Author, Book, Chapter, Publisher, Review, Tag


@pytest.fixture
def sample_data(db):
    """Create sample data for testing."""
    # Publishers
    publisher1 = Publisher.objects.create(name="Tech Books Inc", country="USA")
    publisher2 = Publisher.objects.create(name="Science Press", country="UK")

    # Authors
    author1 = Author.objects.create(name="John Doe", email="john@example.com")
    author2 = Author.objects.create(name="Jane Smith", email="jane@example.com")
    author3 = Author.objects.create(name="Bob Wilson", email="bob@example.com")

    # Tags
    tag_python = Tag.objects.create(name="Python")
    tag_django = Tag.objects.create(name="Django")
    tag_web = Tag.objects.create(name="Web")

    # Book 1 - multiple authors, multiple tags, multiple chapters
    book1 = Book.objects.create(
        title="Django for Beginners",
        isbn="1234567890123",
        price=Decimal("29.99"),
        published_date=date(2024, 1, 15),
        publisher=publisher1,
    )
    book1.authors.add(author1, author2)
    book1.tags.add(tag_python, tag_django)

    Chapter.objects.create(title="Introduction", number=1, page_count=20, book=book1)
    Chapter.objects.create(title="Models", number=2, page_count=35, book=book1)
    Chapter.objects.create(title="Views", number=3, page_count=40, book=book1)

    Review.objects.create(rating=5, comment="Excellent book!", reviewer_name="Alice", book=book1)
    Review.objects.create(rating=4, comment="Very good", reviewer_name="Charlie", book=book1)

    # Book 2 - single author, different tags
    book2 = Book.objects.create(
        title="Advanced Python",
        isbn="1234567890124",
        price=Decimal("49.99"),
        published_date=date(2024, 6, 1),
        publisher=publisher1,
    )
    book2.authors.add(author3)
    book2.tags.add(tag_python, tag_web)

    Chapter.objects.create(title="Metaclasses", number=1, page_count=50, book=book2)
    Chapter.objects.create(title="Descriptors", number=2, page_count=45, book=book2)

    # Book 3 - no chapters, no reviews
    book3 = Book.objects.create(
        title="Web Development Basics",
        isbn="1234567890125",
        price=Decimal("19.99"),
        published_date=date(2023, 3, 10),
        publisher=publisher2,
    )
    book3.authors.add(author1, author3)
    book3.tags.add(tag_web)

    return {
        "publishers": [publisher1, publisher2],
        "authors": [author1, author2, author3],
        "tags": [tag_python, tag_django, tag_web],
        "books": [book1, book2, book3],
    }


class TestPrefetchValuesBasic:
    """Basic tests for prefetch_related().values_nested() functionality."""

    def test_values_nested_without_prefetch_works_normally(self, sample_data):
        """values_nested() without prefetch_related should work as normal Django."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.values_nested("title", "isbn"))

        assert len(result) == 3
        assert all(isinstance(r, dict) for r in result)
        assert all("title" in r and "isbn" in r for r in result)
        assert {"title": "Django for Beginners", "isbn": "1234567890123"} in result

    def test_prefetch_related_without_values_nested_works_normally(self, sample_data):
        """prefetch_related() without values_nested() should work as normal Django."""
        books = list(Book.objects.prefetch_related("authors"))

        assert len(books) == 3
        # Access authors without additional queries (prefetched)
        for book in books:
            list(book.authors.all())


class TestPrefetchValuesManyToMany:
    """Tests for ManyToMany relations with prefetch_related().values_nested()."""

    def test_prefetch_m2m_with_values_returns_nested_list(self, sample_data):
        """Prefetching a M2M relation with values_nested() should return nested list of dicts."""
        from django_nested_values import NestedValuesQuerySet

        # Create a queryset using our custom class
        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors").values_nested("title", "authors"))

        assert len(result) == 3

        # Find Django for Beginners
        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # authors should be a list of dicts
        assert "authors" in django_book
        assert isinstance(django_book["authors"], list)
        assert len(django_book["authors"]) == 2

        # Each author should be a dict with author fields
        author_names = {a["name"] for a in django_book["authors"]}
        assert author_names == {"John Doe", "Jane Smith"}

    def test_prefetch_m2m_with_specific_fields(self, sample_data):
        """Should be able to specify which fields to include from prefetched relation."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        # Only get author name, not email
        result = list(qs.prefetch_related("authors").values_nested("title", "authors__name"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # authors should only have name field
        for author in django_book["authors"]:
            assert "name" in author
            assert "email" not in author

    def test_prefetch_multiple_m2m_relations(self, sample_data):
        """Should support prefetching multiple M2M relations."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors", "tags").values_nested("title", "authors", "tags"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert isinstance(django_book["authors"], list)
        assert isinstance(django_book["tags"], list)
        assert len(django_book["authors"]) == 2
        assert len(django_book["tags"]) == 2

        tag_names = {t["name"] for t in django_book["tags"]}
        assert tag_names == {"Python", "Django"}


class TestPrefetchValuesReverseForeignKey:
    """Tests for reverse ForeignKey (one-to-many) relations."""

    def test_prefetch_reverse_fk_with_values(self, sample_data):
        """Prefetching reverse FK should return nested list of dicts."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("chapters").values_nested("title", "chapters"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert "chapters" in django_book
        assert isinstance(django_book["chapters"], list)
        assert len(django_book["chapters"]) == 3

        # Chapters should be ordered by number
        chapter_titles = [c["title"] for c in django_book["chapters"]]
        assert chapter_titles == ["Introduction", "Models", "Views"]

    def test_prefetch_reverse_fk_with_specific_fields(self, sample_data):
        """Should be able to specify which fields from reverse FK relation."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("chapters").values_nested("title", "chapters__title", "chapters__number"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        for chapter in django_book["chapters"]:
            assert "title" in chapter
            assert "number" in chapter
            assert "page_count" not in chapter

    def test_prefetch_empty_reverse_fk(self, sample_data):
        """Books with no chapters should have empty list."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("chapters").values_nested("title", "chapters"))

        web_book = next(r for r in result if r["title"] == "Web Development Basics")

        assert web_book["chapters"] == []


class TestPrefetchValuesForeignKey:
    """Tests for ForeignKey (many-to-one) relations with select_related behavior."""

    def test_prefetch_fk_with_values(self, sample_data):
        """Prefetching FK should return nested dict (not list)."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("publisher").values_nested("title", "publisher"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # publisher should be a dict, not a list (it's a single object)
        assert "publisher" in django_book
        assert isinstance(django_book["publisher"], dict)
        assert django_book["publisher"]["name"] == "Tech Books Inc"
        assert django_book["publisher"]["country"] == "USA"

    def test_prefetch_fk_with_specific_fields(self, sample_data):
        """Should be able to specify which fields from FK relation."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("publisher").values_nested("title", "publisher__name"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert "name" in django_book["publisher"]
        assert "country" not in django_book["publisher"]


class TestPrefetchValuesNestedRelations:
    """Tests for nested/chained prefetch relations."""

    def test_prefetch_nested_relation(self, sample_data):
        """Should support nested prefetching like authors__books."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Author)
        result = list(qs.prefetch_related("books__chapters").values_nested("name", "books"))

        john = next(r for r in result if r["name"] == "John Doe")

        assert isinstance(john["books"], list)
        # John has 2 books
        assert len(john["books"]) == 2

        # Each book should have chapters nested
        django_book = next(b for b in john["books"] if b["title"] == "Django for Beginners")
        assert "chapters" in django_book
        assert len(django_book["chapters"]) == 3


class TestPrefetchValuesQueryCount:
    """Tests to verify prefetching actually reduces query count."""

    def test_prefetch_reduces_queries(self, sample_data, django_assert_num_queries):
        """Using prefetch_related().values_nested() should use minimal queries."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        # Should be 2 queries: one for books, one for authors (prefetched)
        with django_assert_num_queries(2):
            result = list(qs.prefetch_related("authors").values_nested("title", "authors"))
            # Force evaluation of all authors
            for book in result:
                list(book["authors"])

    def test_multiple_prefetch_query_count(self, sample_data, django_assert_num_queries):
        """Multiple prefetch relations should add one query each."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        # Should be 4 queries: books + authors + tags + chapters
        with django_assert_num_queries(4):
            result = list(
                qs.prefetch_related("authors", "tags", "chapters").values_nested(
                    "title",
                    "authors",
                    "tags",
                    "chapters",
                ),
            )
            for book in result:
                list(book["authors"])
                list(book["tags"])
                list(book["chapters"])


class TestPrefetchValuesWithPrefetchObject:
    """Tests for using Prefetch objects with custom querysets."""

    def test_prefetch_object_with_queryset(self, sample_data):
        """Should support Prefetch objects with custom querysets."""
        from django.db.models import Prefetch

        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        # Only prefetch chapters with page_count > 30
        prefetch = Prefetch("chapters", queryset=Chapter.objects.filter(page_count__gt=30))

        result = list(qs.prefetch_related(prefetch).values_nested("title", "chapters"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # Only chapters with page_count > 30 (Models: 35, Views: 40)
        assert len(django_book["chapters"]) == 2
        chapter_titles = {c["title"] for c in django_book["chapters"]}
        assert chapter_titles == {"Models", "Views"}

    def test_prefetch_object_with_to_attr(self, sample_data):
        """Should support Prefetch objects with to_attr."""
        from django.db.models import Prefetch

        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        prefetch = Prefetch("chapters", queryset=Chapter.objects.filter(number=1), to_attr="first_chapter")

        result = list(qs.prefetch_related(prefetch).values_nested("title", "first_chapter"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # first_chapter should be the result
        assert "first_chapter" in django_book
        assert isinstance(django_book["first_chapter"], list)
        assert len(django_book["first_chapter"]) == 1
        assert django_book["first_chapter"][0]["title"] == "Introduction"


class TestPrefetchValuesEdgeCases:
    """Edge cases and error handling."""

    def test_empty_queryset(self, db):
        """Empty queryset should return empty list."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors").values_nested("title", "authors"))

        assert result == []

    def test_values_with_non_prefetched_relation_raises_error(self, sample_data):
        """Requesting a relation in values_nested() without prefetching should raise error."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        # authors is in values_nested() but not prefetched - should raise
        with pytest.raises(ValueError, match="not prefetched"):
            list(qs.values_nested("title", "authors"))

    def test_chaining_filter_with_prefetch_values(self, sample_data):
        """Should work with filter() in the chain."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(
            qs.filter(price__gt=Decimal("25.00"))
            .prefetch_related("authors")
            .values_nested("title", "price", "authors"),
        )

        # Only books with price > 25.00
        assert len(result) == 2
        titles = {r["title"] for r in result}
        assert titles == {"Django for Beginners", "Advanced Python"}

    def test_ordering_preserved(self, sample_data):
        """QuerySet ordering should be preserved."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.order_by("-price").prefetch_related("authors").values_nested("title", "price", "authors"))

        prices = [r["price"] for r in result]
        assert prices == sorted(prices, reverse=True)

    def test_slicing_queryset(self, sample_data):
        """Slicing should work with prefetch_related().values_nested()."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.order_by("title").prefetch_related("authors").values_nested("title", "authors")[:2])

        assert len(result) == 2
        # First two books alphabetically
        assert result[0]["title"] == "Advanced Python"
        assert result[1]["title"] == "Django for Beginners"

    def test_single_object_with_first(self, sample_data):
        """first() should work with prefetch_related().values_nested()."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = qs.order_by("title").prefetch_related("authors").values_nested("title", "authors").first()

        assert result is not None
        assert result["title"] == "Advanced Python"
        assert isinstance(result["authors"], list)

    def test_count_does_not_evaluate_prefetch(self, sample_data, django_assert_num_queries):
        """count() should not trigger prefetch queries."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        # count() should be 1 query only
        with django_assert_num_queries(1):
            count = qs.prefetch_related("authors").count()

        assert count == 3


class TestPrefetchValuesReverseManyToMany:
    """Tests for reverse ManyToMany relations (from the 'related' side)."""

    def test_reverse_m2m_with_values(self, sample_data):
        """Reverse M2M (Author.books) should return nested list of dicts."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Author)
        result = list(qs.prefetch_related("books").values_nested("name", "books"))

        john = next(r for r in result if r["name"] == "John Doe")

        assert "books" in john
        assert isinstance(john["books"], list)
        assert len(john["books"]) == 2

        book_titles = {b["title"] for b in john["books"]}
        assert book_titles == {"Django for Beginners", "Web Development Basics"}

    def test_reverse_m2m_with_specific_fields(self, sample_data):
        """Should be able to specify fields from reverse M2M."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Author)
        result = list(qs.prefetch_related("books").values_nested("name", "books__title"))

        john = next(r for r in result if r["name"] == "John Doe")

        for book in john["books"]:
            assert "title" in book
            assert "isbn" not in book
            assert "price" not in book


class TestPrefetchValuesDataTypes:
    """Tests to verify data types are preserved correctly."""

    def test_decimal_fields_preserved(self, sample_data):
        """Decimal fields should remain as Decimal type."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors").values_nested("title", "price", "authors"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        assert isinstance(django_book["price"], Decimal)
        assert django_book["price"] == Decimal("29.99")

    def test_date_fields_preserved(self, sample_data):
        """Date fields should remain as date type."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors").values_nested("title", "published_date", "authors"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        assert isinstance(django_book["published_date"], date)
        assert django_book["published_date"] == date(2024, 1, 15)

    def test_integer_fields_in_related(self, sample_data):
        """Integer fields in related objects should be preserved."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("chapters").values_nested("title", "chapters"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        for chapter in django_book["chapters"]:
            assert isinstance(chapter["number"], int)
            assert isinstance(chapter["page_count"], int)


class TestPrefetchValuesMixedRelations:
    """Tests for combining different relation types."""

    def test_m2m_and_fk_together(self, sample_data):
        """Should handle M2M and FK prefetch together."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors", "publisher").values_nested("title", "authors", "publisher"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # M2M returns list
        assert isinstance(django_book["authors"], list)
        assert len(django_book["authors"]) == 2

        # FK returns dict
        assert isinstance(django_book["publisher"], dict)
        assert django_book["publisher"]["name"] == "Tech Books Inc"

    def test_m2m_and_reverse_fk_together(self, sample_data):
        """Should handle M2M and reverse FK prefetch together."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors", "chapters").values_nested("title", "authors", "chapters"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert isinstance(django_book["authors"], list)
        assert len(django_book["authors"]) == 2

        assert isinstance(django_book["chapters"], list)
        assert len(django_book["chapters"]) == 3

    def test_all_relation_types_together(self, sample_data):
        """Should handle all relation types in one query."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(
            qs.prefetch_related("authors", "tags", "chapters", "reviews", "publisher").values_nested(
                "title",
                "authors",
                "tags",
                "chapters",
                "reviews",
                "publisher",
            ),
        )

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert isinstance(django_book["authors"], list)
        assert isinstance(django_book["tags"], list)
        assert isinstance(django_book["chapters"], list)
        assert isinstance(django_book["reviews"], list)
        assert isinstance(django_book["publisher"], dict)

        assert len(django_book["authors"]) == 2
        assert len(django_book["tags"]) == 2
        assert len(django_book["chapters"]) == 3
        assert len(django_book["reviews"]) == 2


class TestPrefetchValuesWithFilters:
    """Tests for queryset filtering combined with prefetch."""

    def test_filter_on_main_model(self, sample_data):
        """Filter on the main model should work."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.filter(publisher__country="USA").prefetch_related("authors").values_nested("title", "authors"))

        # Only books from USA publisher
        assert len(result) == 2
        titles = {r["title"] for r in result}
        assert titles == {"Django for Beginners", "Advanced Python"}

    def test_filter_with_related_lookup(self, sample_data):
        """Filter using related lookup should work."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.filter(authors__name="John Doe").prefetch_related("authors").values_nested("title", "authors"))

        # Books where John Doe is an author
        assert len(result) == 2
        titles = {r["title"] for r in result}
        assert titles == {"Django for Beginners", "Web Development Basics"}

    def test_exclude_works(self, sample_data):
        """exclude() should work with prefetch values."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.exclude(title="Advanced Python").prefetch_related("authors").values_nested("title", "authors"))

        assert len(result) == 2
        titles = {r["title"] for r in result}
        assert "Advanced Python" not in titles


class TestPrefetchValuesIdField:
    """Tests for handling the id/pk field correctly."""

    def test_id_included_when_requested(self, sample_data):
        """id field should be included when explicitly requested."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors").values_nested("id", "title", "authors"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        assert "id" in django_book
        assert isinstance(django_book["id"], int)

    def test_id_not_included_when_not_requested(self, sample_data):
        """id field should not be included when not requested."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors").values_nested("title", "authors"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        assert "id" not in django_book

    def test_related_id_handling(self, sample_data):
        """Related object ids should be included by default (all fields)."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors").values_nested("title", "authors"))

        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        # When no specific fields requested, all fields including id are returned
        for author in django_book["authors"]:
            assert "id" in author


class TestPrefetchValuesEmptyRelations:
    """Tests for handling empty relations."""

    def test_book_with_no_authors(self, db):
        """Book with no authors should have empty authors list."""
        from django_nested_values import NestedValuesQuerySet

        publisher = Publisher.objects.create(name="Test", country="US")
        Book.objects.create(
            title="Orphan Book",
            isbn="0000000000000",
            price=Decimal("10.00"),
            published_date=date(2024, 1, 1),
            publisher=publisher,
        )

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors").values_nested("title", "authors"))

        orphan = next(r for r in result if r["title"] == "Orphan Book")
        assert orphan["authors"] == []

    def test_author_with_no_books(self, db):
        """Author with no books should have empty books list."""
        from django_nested_values import NestedValuesQuerySet

        Author.objects.create(name="Lonely Author", email="lonely@example.com")

        qs = NestedValuesQuerySet(model=Author)
        result = list(qs.prefetch_related("books").values_nested("name", "books"))

        lonely = next(r for r in result if r["name"] == "Lonely Author")
        assert lonely["books"] == []


class TestPrefetchValuesWithManager:
    """Tests for using NestedValuesQuerySet as a manager."""

    def test_as_manager(self, sample_data):
        """Should work when used as a custom manager."""
        from django.db import models

        from django_nested_values import NestedValuesQuerySet

        # Create a manager from the queryset
        CustomManager = models.Manager.from_queryset(NestedValuesQuerySet)

        # Use it directly (simulating what happens when attached to a model)
        manager = CustomManager()
        manager.model = Book
        manager._db = None

        qs = manager.get_queryset()
        result = list(qs.prefetch_related("authors").values_nested("title", "authors"))

        assert len(result) == 3
        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        assert len(django_book["authors"]) == 2
