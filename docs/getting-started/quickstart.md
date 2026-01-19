# Quick Start

## Setup

Add `NestedValuesQuerySet` as your model's manager:

```python
from django.db import models
from django_nested_values import NestedValuesQuerySet


class Publisher(models.Model):
    name = models.CharField(max_length=100)

    objects = NestedValuesQuerySet.as_manager()


class Author(models.Model):
    name = models.CharField(max_length=100)

    objects = NestedValuesQuerySet.as_manager()


class Book(models.Model):
    title = models.CharField(max_length=200)
    publisher = models.ForeignKey(Publisher, on_delete=models.CASCADE)
    authors = models.ManyToManyField(Author, related_name="books")

    objects = NestedValuesQuerySet.as_manager()
```

## Basic Usage

```python
# Get all fields as nested dicts
books = Book.objects.select_related("publisher").prefetch_related("authors").values_nested()
# [{"id": 1, "title": "...", "publisher": {...}, "authors": [...]}, ...]

# Control which fields with only()
books = Book.objects.only("title").prefetch_related("authors").values_nested()
# [{"id": 1, "title": "...", "authors": [...]}, ...]
```

## ForeignKey: select_related vs prefetch_related

For ForeignKey relations, use `select_related()` for efficient JOINs:

```python
# Efficient: 1 query with JOIN
books = Book.objects.select_related("publisher").values_nested()

# Less efficient: 2 queries (books + publishers)
books = Book.objects.prefetch_related("publisher").values_nested()
```

Both return the same nested structure:
```python
{"id": 1, "title": "...", "publisher": {"id": 1, "name": "...", "country": "..."}}
```

## ManyToMany and Reverse ForeignKey

For M2M and reverse FK, use `prefetch_related()`:

```python
# ManyToMany
books = Book.objects.prefetch_related("authors").values_nested()
# {"id": 1, "title": "...", "authors": [{"id": 1, "name": "..."}, ...]}

# Reverse ForeignKey
books = Book.objects.prefetch_related("chapters").values_nested()
# {"id": 1, "title": "...", "chapters": [{"id": 1, "title": "Chapter 1"}, ...]}
```

## Controlling Related Fields

Use `Prefetch` objects with `only()` to control which fields are included:

```python
from django.db.models import Prefetch

# Only fetch author names
books = Book.objects.only("title").prefetch_related(
    Prefetch("authors", queryset=Author.objects.only("name"))
).values_nested()
# {"id": 1, "title": "...", "authors": [{"id": 1, "name": "..."}]}

# Filter prefetched data
books = Book.objects.only("title").prefetch_related(
    Prefetch("authors", queryset=Author.objects.filter(name__startswith="A"))
).values_nested()

# Custom to_attr
books = Book.objects.only("title").prefetch_related(
    Prefetch("chapters", queryset=Chapter.objects.filter(number=1), to_attr="first_chapter")
).values_nested()
# {"id": 1, "title": "...", "first_chapter": [{"id": 1, "title": "Introduction"}]}
```

## Combining Relations

Mix `select_related()` and `prefetch_related()`:

```python
books = (
    Book.objects
    .only("title")
    .select_related("publisher")           # FK: 1 query with JOIN
    .prefetch_related("authors", "tags")   # M2M: +2 queries
    .values_nested()
)
# Total: 3 queries
# {"id": 1, "title": "...", "publisher": {...}, "authors": [...], "tags": [...]}
```

## Use Case: API Endpoints

The main use case is APIs where data gets passed to Pydantic models:

| Approach | Flow |
|----------|------|
| Standard Django | DB → Django Model → dict → Pydantic |
| django-nested-values | DB → dict → Pydantic |

```python
from ninja import NinjaAPI

@api.get("/books", response=list[BookSchema])
def list_books(request):
    return list(
        Book.objects
        .only("id", "title")
        .select_related("publisher")
        .prefetch_related("authors")
        .values_nested()
    )
```

## Benchmark

The included benchmark (`benchmarks/benchmark.py`) tests fetching 1000 books with multiple relations. Both approaches use the same number of database queries.

```bash
uv run python benchmarks/benchmark.py
```
