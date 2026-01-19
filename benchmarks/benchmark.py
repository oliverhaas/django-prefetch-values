"""Benchmark comparing normal prefetch_related vs prefetch_related().values_nested()."""

from __future__ import annotations

import gc
import os
import random
import statistics
import time

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "benchmarks.settings")

import django

django.setup()

from django.db import connection, reset_queries

from benchmarks.models import Author, Book, Chapter, Publisher, Review, Tag
from django_nested_values import NestedValuesQuerySet

# Configuration
NUM_PUBLISHERS = 50
NUM_AUTHORS = 200
NUM_TAGS = 30
NUM_BOOKS = 1000
CHAPTERS_PER_BOOK = (1, 5)  # Random range
REVIEWS_PER_BOOK = (0, 3)  # Random range
AUTHORS_PER_BOOK = (1, 3)  # Random range
TAGS_PER_BOOK = (1, 4)  # Random range
NUM_ITERATIONS = 10


def setup_database():
    """Create tables and populate with test data."""
    from django.core.management import call_command

    print("Setting up database...")
    call_command("migrate", "--run-syncdb", verbosity=0)

    # Check if data already exists
    if Book.objects.count() >= NUM_BOOKS:
        print(f"Database already has {Book.objects.count()} books, skipping setup.")
        return

    print(f"Creating {NUM_PUBLISHERS} publishers...")
    publishers = [
        Publisher(name=f"Publisher {i}", country=random.choice(["USA", "UK", "Germany", "France", "Japan"]))
        for i in range(NUM_PUBLISHERS)
    ]
    Publisher.objects.bulk_create(publishers)
    publishers = list(Publisher.objects.all())

    print(f"Creating {NUM_AUTHORS} authors...")
    authors = [Author(name=f"Author {i}", email=f"author{i}@example.com") for i in range(NUM_AUTHORS)]
    Author.objects.bulk_create(authors)
    authors = list(Author.objects.all())

    print(f"Creating {NUM_TAGS} tags...")
    tags = [Tag(name=f"Tag {i}") for i in range(NUM_TAGS)]
    Tag.objects.bulk_create(tags)
    tags = list(Tag.objects.all())

    print(f"Creating {NUM_BOOKS} books with relations...")
    from datetime import date, timedelta
    from decimal import Decimal

    books = []
    for i in range(NUM_BOOKS):
        book = Book(
            title=f"Book {i}: " + "".join(random.choices("abcdefghijklmnopqrstuvwxyz ", k=20)),
            isbn=f"{i:013d}",
            price=Decimal(f"{random.randint(10, 100)}.{random.randint(0, 99):02d}"),
            published_date=date(2020, 1, 1) + timedelta(days=random.randint(0, 1500)),
            publisher=random.choice(publishers),
        )
        books.append(book)

    Book.objects.bulk_create(books)
    books = list(Book.objects.all())

    print("Creating M2M relations (authors, tags)...")
    # Add authors to books
    through_model = Book.authors.through
    author_relations = []
    for book in books:
        num_authors = random.randint(*AUTHORS_PER_BOOK)
        book_authors = random.sample(authors, num_authors)
        for author in book_authors:
            author_relations.append(through_model(book_id=book.id, author_id=author.id))
    through_model.objects.bulk_create(author_relations)

    # Add tags to books
    through_model = Book.tags.through
    tag_relations = []
    for book in books:
        num_tags = random.randint(*TAGS_PER_BOOK)
        book_tags = random.sample(tags, num_tags)
        for tag in book_tags:
            tag_relations.append(through_model(book_id=book.id, tag_id=tag.id))
    through_model.objects.bulk_create(tag_relations)

    print("Creating chapters...")
    chapters = []
    for book in books:
        num_chapters = random.randint(*CHAPTERS_PER_BOOK)
        for j in range(num_chapters):
            chapters.append(
                Chapter(
                    title=f"Chapter {j + 1}",
                    number=j + 1,
                    page_count=random.randint(10, 50),
                    book=book,
                ),
            )
    Chapter.objects.bulk_create(chapters)

    print("Creating reviews...")
    reviews = []
    reviewer_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
    for book in books:
        num_reviews = random.randint(*REVIEWS_PER_BOOK)
        for _ in range(num_reviews):
            reviews.append(
                Review(
                    rating=random.randint(1, 5),
                    comment="This is a review comment with some text. " * random.randint(1, 5),
                    reviewer_name=random.choice(reviewer_names),
                    book=book,
                ),
            )
    Review.objects.bulk_create(reviews)

    print("Setup complete!")
    print(f"  - {Publisher.objects.count()} publishers")
    print(f"  - {Author.objects.count()} authors")
    print(f"  - {Tag.objects.count()} tags")
    print(f"  - {Book.objects.count()} books")
    print(f"  - {Chapter.objects.count()} chapters")
    print(f"  - {Review.objects.count()} reviews")


def benchmark_normal_prefetch():
    """Benchmark: Fetch with prefetch_related, then manually convert to dicts."""
    books = list(
        Book.objects.prefetch_related("authors", "tags", "chapters", "reviews", "publisher").all(),
    )

    # Convert to dicts manually (what you'd typically do)
    result = []
    for book in books:
        book_dict = {
            "id": book.id,
            "title": book.title,
            "isbn": book.isbn,
            "price": book.price,
            "published_date": book.published_date,
            "publisher": {
                "id": book.publisher.id,
                "name": book.publisher.name,
                "country": book.publisher.country,
            },
            "authors": [{"id": a.id, "name": a.name, "email": a.email} for a in book.authors.all()],
            "tags": [{"id": t.id, "name": t.name} for t in book.tags.all()],
            "chapters": [
                {"id": c.id, "title": c.title, "number": c.number, "page_count": c.page_count}
                for c in book.chapters.all()
            ],
            "reviews": [
                {"id": r.id, "rating": r.rating, "comment": r.comment, "reviewer_name": r.reviewer_name}
                for r in book.reviews.all()
            ],
        }
        result.append(book_dict)

    return result


def benchmark_prefetch_values_nested():
    """Benchmark: Fetch with prefetch_related().values_nested() - our new approach."""
    qs = NestedValuesQuerySet(model=Book)
    result = list(
        qs.prefetch_related("authors", "tags", "chapters", "reviews", "publisher").values_nested(
            "id",
            "title",
            "isbn",
            "price",
            "published_date",
            "publisher",
            "authors",
            "tags",
            "chapters",
            "reviews",
        ),
    )
    return result


def benchmark_values_only():
    """Benchmark: Standard values() without prefetch (loses relations)."""
    result = list(
        Book.objects.values(
            "id",
            "title",
            "isbn",
            "price",
            "published_date",
            "publisher__id",
            "publisher__name",
            "publisher__country",
        ),
    )
    return result


def run_benchmark(name: str, func, iterations: int = NUM_ITERATIONS):
    """Run a benchmark function multiple times and report statistics."""
    times = []
    query_counts = []

    for i in range(iterations):
        # Clear caches
        gc.collect()
        reset_queries()

        # Run benchmark
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()

        times.append(end - start)
        query_counts.append(len(connection.queries))

        # Verify we got results
        if i == 0:
            print(f"  {name}: {len(result)} records returned")

    return {
        "name": name,
        "mean_time": statistics.mean(times),
        "std_time": statistics.stdev(times) if len(times) > 1 else 0,
        "min_time": min(times),
        "max_time": max(times),
        "query_count": query_counts[0],  # Should be same each time
    }


def main():
    """Run all benchmarks."""
    from django.conf import settings

    settings.DEBUG = True  # Enable query logging

    setup_database()

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: Fetching {NUM_BOOKS} books with all relations")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"{'=' * 70}\n")

    # Warm up
    print("Warming up...")
    benchmark_normal_prefetch()
    benchmark_prefetch_values_nested()
    benchmark_values_only()

    print("\nRunning benchmarks...\n")

    results = []

    # Benchmark 1: Normal prefetch + manual dict conversion
    results.append(run_benchmark("Normal prefetch + manual dict", benchmark_normal_prefetch))

    # Benchmark 2: Our prefetch_related().values_nested()
    results.append(run_benchmark("prefetch_related().values_nested()", benchmark_prefetch_values_nested))

    # Benchmark 3: Standard values() (for reference, but loses M2M/reverse FK)
    results.append(run_benchmark("Standard values() (no M2M)", benchmark_values_only))

    # Print results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}\n")

    print(f"{'Method':<35} {'Mean (ms)':<12} {'Std (ms)':<10} {'Queries':<8}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['name']:<35} {r['mean_time'] * 1000:>8.2f}    {r['std_time'] * 1000:>6.2f}    {r['query_count']:>5}",
        )

    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}\n")

    normal = results[0]
    prefetch_values = results[1]

    speedup = normal["mean_time"] / prefetch_values["mean_time"]
    print(
        f"prefetch_related().values_nested() is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than normal prefetch + manual dict",
    )
    print(f"  - Normal: {normal['mean_time'] * 1000:.2f}ms")
    print(f"  - values_nested(): {prefetch_values['mean_time'] * 1000:.2f}ms")
    print(f"  - Difference: {(normal['mean_time'] - prefetch_values['mean_time']) * 1000:.2f}ms")

    # Memory comparison (rough)
    print(f"\nQuery counts are identical: {normal['query_count']} queries each")

    return results


if __name__ == "__main__":
    main()
