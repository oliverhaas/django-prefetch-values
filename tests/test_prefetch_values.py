"""TDD tests for values_nested() functionality.

These tests define the expected behavior of the new API:
- values_nested() takes no arguments
- Use .only() for field selection
- Use .select_related() for ForeignKey (1 query with JOIN)
- Use .prefetch_related() for ManyToMany and reverse ForeignKey
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from django.db.models import Prefetch

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


class TestValuesNestedBasic:
    """Basic tests for values_nested() without relations."""

    def test_values_nested_all_fields(self, sample_data):
        """values_nested() without only() returns all fields."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.values_nested())

        assert len(result) == 3
        assert all(isinstance(r, dict) for r in result)
        # Should have all concrete fields
        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        assert "id" in django_book
        assert "title" in django_book
        assert "isbn" in django_book
        assert "price" in django_book
        assert "published_date" in django_book
        assert "publisher_id" in django_book

    def test_values_nested_with_only(self, sample_data):
        """values_nested() with only() returns only specified fields."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.only("title", "isbn").values_nested())

        assert len(result) == 3
        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        assert "title" in django_book
        assert "isbn" in django_book
        # id is always included by Django's only()
        assert "id" in django_book
        # These should not be present
        assert "price" not in django_book
        assert "published_date" not in django_book


class TestSelectRelated:
    """Tests for ForeignKey relations using select_related()."""

    def test_select_related_fk_returns_nested_dict(self, sample_data):
        """select_related() FK should return nested dict (not list)."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.select_related("publisher").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # publisher should be a dict, not a list
        assert "publisher" in django_book
        assert isinstance(django_book["publisher"], dict)
        assert django_book["publisher"]["name"] == "Tech Books Inc"
        assert django_book["publisher"]["country"] == "USA"

    def test_select_related_with_only_on_main(self, sample_data):
        """select_related() with only() on main model."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.only("title").select_related("publisher").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert "title" in django_book
        assert "publisher" in django_book
        assert isinstance(django_book["publisher"], dict)
        # publisher should have all its fields
        assert "name" in django_book["publisher"]
        assert "country" in django_book["publisher"]

    def test_select_related_with_only_on_relation(self, sample_data):
        """select_related() with only() specifying relation fields."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        # only() can specify related fields with double-underscore
        result = list(qs.only("title", "publisher__name").select_related("publisher").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert "title" in django_book
        assert "publisher" in django_book
        assert "name" in django_book["publisher"]
        # country should not be present since we only asked for publisher__name
        assert "country" not in django_book["publisher"]

    def test_select_related_query_count(self, sample_data, django_assert_num_queries):
        """select_related() should use 1 query (JOIN)."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        # Should be 1 query with JOIN
        with django_assert_num_queries(1):
            result = list(qs.select_related("publisher").values_nested())
            # Access publisher to ensure it's loaded
            for book in result:
                _ = book["publisher"]


class TestPrefetchRelatedManyToMany:
    """Tests for ManyToMany relations using prefetch_related()."""

    def test_prefetch_m2m_returns_nested_list(self, sample_data):
        """prefetch_related() M2M should return nested list of dicts."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert "authors" in django_book
        assert isinstance(django_book["authors"], list)
        assert len(django_book["authors"]) == 2

        author_names = {a["name"] for a in django_book["authors"]}
        assert author_names == {"John Doe", "Jane Smith"}

    def test_prefetch_m2m_with_only_on_main(self, sample_data):
        """prefetch_related() with only() on main model."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.only("title").prefetch_related("authors").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert "title" in django_book
        assert "authors" in django_book
        # price should not be present
        assert "price" not in django_book

    def test_prefetch_m2m_with_prefetch_object_only(self, sample_data):
        """Prefetch object with only() on related queryset."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(
            qs.only("title")
            .prefetch_related(Prefetch("authors", queryset=Author.objects.only("name")))
            .values_nested(),
        )

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # authors should only have name (and id from only())
        for author in django_book["authors"]:
            assert "name" in author
            assert "id" in author
            assert "email" not in author

    def test_prefetch_multiple_m2m(self, sample_data):
        """Multiple M2M relations with prefetch_related()."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.only("title").prefetch_related("authors", "tags").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert isinstance(django_book["authors"], list)
        assert isinstance(django_book["tags"], list)
        assert len(django_book["authors"]) == 2
        assert len(django_book["tags"]) == 2

        tag_names = {t["name"] for t in django_book["tags"]}
        assert tag_names == {"Python", "Django"}

    def test_prefetch_m2m_query_count(self, sample_data, django_assert_num_queries):
        """prefetch_related() M2M should use 2 queries."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        # Should be 2 queries: books + authors
        with django_assert_num_queries(2):
            result = list(qs.prefetch_related("authors").values_nested())
            for book in result:
                list(book["authors"])


class TestPrefetchRelatedReverseForeignKey:
    """Tests for reverse ForeignKey (one-to-many) relations."""

    def test_prefetch_reverse_fk_returns_nested_list(self, sample_data):
        """prefetch_related() reverse FK should return nested list of dicts."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("chapters").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert "chapters" in django_book
        assert isinstance(django_book["chapters"], list)
        assert len(django_book["chapters"]) == 3

        # Chapters should be ordered by number (model has ordering)
        chapter_titles = [c["title"] for c in django_book["chapters"]]
        assert chapter_titles == ["Introduction", "Models", "Views"]

    def test_prefetch_reverse_fk_with_prefetch_object_only(self, sample_data):
        """Prefetch object with only() on reverse FK queryset."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(
            qs.only("title")
            .prefetch_related(Prefetch("chapters", queryset=Chapter.objects.only("title", "number")))
            .values_nested(),
        )

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        for chapter in django_book["chapters"]:
            assert "title" in chapter
            assert "number" in chapter
            assert "page_count" not in chapter

    def test_prefetch_empty_reverse_fk(self, sample_data):
        """Books with no chapters should have empty list."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.only("title").prefetch_related("chapters").values_nested())

        web_book = next(r for r in result if r["title"] == "Web Development Basics")
        assert web_book["chapters"] == []


class TestPrefetchRelatedForeignKey:
    """Tests for ForeignKey using prefetch_related() (less efficient than select_related)."""

    def test_prefetch_fk_returns_nested_dict(self, sample_data):
        """prefetch_related() FK should return nested dict (not list)."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("publisher").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # publisher should be a dict, not a list
        assert "publisher" in django_book
        assert isinstance(django_book["publisher"], dict)
        assert django_book["publisher"]["name"] == "Tech Books Inc"

    def test_prefetch_fk_query_count(self, sample_data, django_assert_num_queries):
        """prefetch_related() FK should use 2 queries (less efficient than select_related)."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        # Should be 2 queries: books + publishers
        with django_assert_num_queries(2):
            result = list(qs.prefetch_related("publisher").values_nested())
            for book in result:
                _ = book["publisher"]


class TestReverseManyToMany:
    """Tests for reverse ManyToMany relations."""

    def test_reverse_m2m_returns_nested_list(self, sample_data):
        """Reverse M2M (Author.books) should return nested list of dicts."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Author)
        result = list(qs.prefetch_related("books").values_nested())

        john = next(r for r in result if r["name"] == "John Doe")

        assert "books" in john
        assert isinstance(john["books"], list)
        assert len(john["books"]) == 2

        book_titles = {b["title"] for b in john["books"]}
        assert book_titles == {"Django for Beginners", "Web Development Basics"}


class TestNestedPrefetch:
    """Tests for nested/chained prefetch relations."""

    def test_nested_prefetch(self, sample_data):
        """Should support nested prefetching like books__chapters."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Author)
        result = list(qs.only("name").prefetch_related("books__chapters").values_nested())

        john = next(r for r in result if r["name"] == "John Doe")

        assert isinstance(john["books"], list)
        assert len(john["books"]) == 2

        django_book = next(b for b in john["books"] if b["title"] == "Django for Beginners")
        assert "chapters" in django_book
        assert len(django_book["chapters"]) == 3


class TestCombinedSelectAndPrefetch:
    """Tests for combining select_related and prefetch_related."""

    def test_select_related_and_prefetch_related_together(self, sample_data):
        """Should handle both select_related and prefetch_related."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.only("title").select_related("publisher").prefetch_related("authors").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # FK via select_related returns dict
        assert isinstance(django_book["publisher"], dict)
        assert django_book["publisher"]["name"] == "Tech Books Inc"

        # M2M via prefetch_related returns list
        assert isinstance(django_book["authors"], list)
        assert len(django_book["authors"]) == 2

    def test_combined_query_count(self, sample_data, django_assert_num_queries):
        """Combined should use 1 (JOIN for FK) + 1 (prefetch for M2M) = 2 queries."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        with django_assert_num_queries(2):
            result = list(qs.select_related("publisher").prefetch_related("authors").values_nested())
            for book in result:
                _ = book["publisher"]
                list(book["authors"])

    def test_all_relation_types_together(self, sample_data):
        """Should handle all relation types in one query."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(
            qs.only("title")
            .select_related("publisher")
            .prefetch_related("authors", "tags", "chapters", "reviews")
            .values_nested(),
        )

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert isinstance(django_book["publisher"], dict)
        assert isinstance(django_book["authors"], list)
        assert isinstance(django_book["tags"], list)
        assert isinstance(django_book["chapters"], list)
        assert isinstance(django_book["reviews"], list)

        assert len(django_book["authors"]) == 2
        assert len(django_book["tags"]) == 2
        assert len(django_book["chapters"]) == 3
        assert len(django_book["reviews"]) == 2


class TestPrefetchObject:
    """Tests for using Prefetch objects with custom querysets."""

    def test_prefetch_object_with_filter(self, sample_data):
        """Prefetch objects with filtered querysets."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        prefetch = Prefetch("chapters", queryset=Chapter.objects.filter(page_count__gt=30))
        result = list(qs.only("title").prefetch_related(prefetch).values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        # Only chapters with page_count > 30 (Models: 35, Views: 40)
        assert len(django_book["chapters"]) == 2
        chapter_titles = {c["title"] for c in django_book["chapters"]}
        assert chapter_titles == {"Models", "Views"}

    def test_prefetch_object_with_to_attr(self, sample_data):
        """Prefetch objects with to_attr."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        prefetch = Prefetch("chapters", queryset=Chapter.objects.filter(number=1), to_attr="first_chapter")
        result = list(qs.only("title").prefetch_related(prefetch).values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")

        assert "first_chapter" in django_book
        assert isinstance(django_book["first_chapter"], list)
        assert len(django_book["first_chapter"]) == 1
        assert django_book["first_chapter"][0]["title"] == "Introduction"


class TestQuerySetChaining:
    """Tests for queryset chaining with filter, exclude, order_by, etc."""

    def test_filter_on_main_model(self, sample_data):
        """filter() should work with values_nested()."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.filter(publisher__country="USA").only("title").prefetch_related("authors").values_nested())

        assert len(result) == 2
        titles = {r["title"] for r in result}
        assert titles == {"Django for Beginners", "Advanced Python"}

    def test_exclude_works(self, sample_data):
        """exclude() should work with values_nested()."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.exclude(title="Advanced Python").only("title").prefetch_related("authors").values_nested())

        assert len(result) == 2
        titles = {r["title"] for r in result}
        assert "Advanced Python" not in titles

    def test_ordering_preserved(self, sample_data):
        """order_by() should be preserved."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.order_by("-price").only("title", "price").values_nested())

        prices = [r["price"] for r in result]
        assert prices == sorted(prices, reverse=True)

    def test_slicing_works(self, sample_data):
        """Slicing should work with values_nested()."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.order_by("title").only("title").prefetch_related("authors").values_nested()[:2])

        assert len(result) == 2
        assert result[0]["title"] == "Advanced Python"
        assert result[1]["title"] == "Django for Beginners"

    def test_first_works(self, sample_data):
        """first() should work with values_nested()."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = qs.order_by("title").only("title").prefetch_related("authors").values_nested().first()

        assert result is not None
        assert result["title"] == "Advanced Python"
        assert isinstance(result["authors"], list)


class TestDataTypes:
    """Tests to verify data types are preserved correctly."""

    def test_decimal_fields_preserved(self, sample_data):
        """Decimal fields should remain as Decimal type."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.only("title", "price").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        assert isinstance(django_book["price"], Decimal)
        assert django_book["price"] == Decimal("29.99")

    def test_date_fields_preserved(self, sample_data):
        """Date fields should remain as date type."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.only("title", "published_date").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        assert isinstance(django_book["published_date"], date)
        assert django_book["published_date"] == date(2024, 1, 15)

    def test_integer_fields_in_related(self, sample_data):
        """Integer fields in related objects should be preserved."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.only("title").prefetch_related("chapters").values_nested())

        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        for chapter in django_book["chapters"]:
            assert isinstance(chapter["number"], int)
            assert isinstance(chapter["page_count"], int)


class TestEmptyRelations:
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
        result = list(qs.only("title").prefetch_related("authors").values_nested())

        orphan = next(r for r in result if r["title"] == "Orphan Book")
        assert orphan["authors"] == []

    def test_author_with_no_books(self, db):
        """Author with no books should have empty books list."""
        from django_nested_values import NestedValuesQuerySet

        Author.objects.create(name="Lonely Author", email="lonely@example.com")

        qs = NestedValuesQuerySet(model=Author)
        result = list(qs.only("name").prefetch_related("books").values_nested())

        lonely = next(r for r in result if r["name"] == "Lonely Author")
        assert lonely["books"] == []


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_queryset(self, db):
        """Empty queryset should return empty list."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)
        result = list(qs.prefetch_related("authors").values_nested())

        assert result == []

    def test_count_does_not_evaluate_prefetch(self, sample_data, django_assert_num_queries):
        """count() should not trigger prefetch queries."""
        from django_nested_values import NestedValuesQuerySet

        qs = NestedValuesQuerySet(model=Book)

        with django_assert_num_queries(1):
            count = qs.prefetch_related("authors").count()

        assert count == 3


class TestAsManager:
    """Tests for using NestedValuesQuerySet as a manager."""

    def test_as_manager(self, sample_data):
        """Should work when used as a custom manager."""
        from django.db import models

        from django_nested_values import NestedValuesQuerySet

        CustomManager = models.Manager.from_queryset(NestedValuesQuerySet)

        manager = CustomManager()
        manager.model = Book
        manager._db = None

        qs = manager.get_queryset()
        result = list(qs.only("title").prefetch_related("authors").values_nested())

        assert len(result) == 3
        django_book = next(r for r in result if r["title"] == "Django for Beginners")
        assert len(django_book["authors"]) == 2
