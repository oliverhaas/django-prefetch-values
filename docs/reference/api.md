# API Reference

## NestedValuesQuerySet

A custom QuerySet that adds `.values_nested()` to return nested dictionaries with prefetched relations.

Inherits from `django.db.models.QuerySet`.

### Setup

```python
from django_nested_values import NestedValuesQuerySet

class MyModel(models.Model):
    objects = NestedValuesQuerySet.as_manager()
```

### values_nested(*fields, **expressions)

Returns dictionaries with prefetched relations included as nested lists.

```python
Book.objects.prefetch_related("authors").values_nested()
# [{"id": 1, "title": "...", "authors": [{"id": 1, "name": "..."}, ...]}, ...]

Book.objects.prefetch_related("authors").values_nested("id", "title")
# [{"id": 1, "title": "...", "authors": [...]}, ...]
```

## Supported Relations

- **ManyToManyField**
- **Reverse ForeignKey** (ManyToOneRel)
- **Reverse ManyToMany** (ManyToManyRel)
- **Nested Prefetches**

## Query Efficiency

Same query count as standard Django prefetching: one query for the main model plus one per prefetch.

```python
# 2 queries: books + authors
Book.objects.prefetch_related("authors").values_nested()

# 3 queries: publishers + books + authors
Publisher.objects.prefetch_related(
    Prefetch("books", queryset=Book.objects.prefetch_related("authors"))
).values_nested()
```

The performance benefit comes from avoiding Django model instantiation.
