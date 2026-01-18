# Quick Start

## Setup

Add `PrefetchValuesQuerySet` as your model's manager:

```python
from django.db import models
from django_prefetch_values import PrefetchValuesQuerySet


class Author(models.Model):
    name = models.CharField(max_length=100)

    objects = PrefetchValuesQuerySet.as_manager()


class Book(models.Model):
    title = models.CharField(max_length=200)
    authors = models.ManyToManyField(Author, related_name="books")

    objects = PrefetchValuesQuerySet.as_manager()
```

## Usage

```python
# Standard Django - prefetch_related is IGNORED with values()
books = Book.objects.prefetch_related("authors").values()
# [{"id": 1, "title": "Book 1"}, ...]  # No authors!

# With django-prefetch-values - prefetched data is included
books = Book.objects.prefetch_related("authors").values()
# [{"id": 1, "title": "Book 1", "authors": [{"id": 1, "name": "Author 1"}, ...]}, ...]
```

## Supported Relations

- **ManyToMany**: `Book.objects.prefetch_related("authors").values()`
- **Reverse ForeignKey**: `Author.objects.prefetch_related("books").values()`
- **Nested**: `Publisher.objects.prefetch_related(Prefetch("books", queryset=Book.objects.prefetch_related("authors"))).values()`

## Prefetch Objects

```python
from django.db.models import Prefetch

# Filter prefetched data
Book.objects.prefetch_related(
    Prefetch("authors", queryset=Author.objects.filter(name__startswith="A"))
).values()

# Custom to_attr
Book.objects.prefetch_related(
    Prefetch("authors", to_attr="author_list")
).values()
# [{"id": 1, "title": "Book 1", "author_list": [...]}, ...]
```

## Use Case: API Endpoints

The main use case is APIs where data gets passed to Pydantic models. Instead of instantiating Django models just to serialize them, fetch dicts directly:

| Approach | Flow |
|----------|------|
| Standard Django | DB → Django Model → dict → Pydantic |
| django-prefetch-values | DB → dict → Pydantic |

```python
from ninja import NinjaAPI

@api.get("/books", response=list[BookSchema])
def list_books(request):
    return list(Book.objects.prefetch_related("authors").values())
```

## Benchmark

The included benchmark (`benchmarks/benchmark.py`) tests fetching 1000 books, each with:

- 1 publisher (ForeignKey)
- 1-3 authors (ManyToMany)
- 1-4 tags (ManyToMany)
- 1-5 chapters (reverse ForeignKey)
- 0-3 reviews (reverse ForeignKey)

Both approaches use the same number of database queries.

```bash
uv run python benchmarks/benchmark.py
```
