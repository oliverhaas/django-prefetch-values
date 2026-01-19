# API Reference

## NestedValuesQuerySet

A custom QuerySet that adds `.values_nested()` to return nested dictionaries with related objects.

Inherits from `django.db.models.QuerySet`.

### Setup

```python
from django_nested_values import NestedValuesQuerySet

class MyModel(models.Model):
    objects = NestedValuesQuerySet.as_manager()
```

### values_nested()

Returns dictionaries with related objects included as nested dicts/lists.

**Takes no arguments.** Use standard Django methods to control the output:

- `.only()` - Select which fields to include
- `.select_related()` - Include ForeignKey relations (single dict)
- `.prefetch_related()` - Include M2M/reverse FK relations (list of dicts)

```python
# All fields, no relations
Book.objects.values_nested()
# [{"id": 1, "title": "...", "isbn": "...", ...}, ...]

# Selected fields with relations
Book.objects.only("title").select_related("publisher").prefetch_related("authors").values_nested()
# [{"id": 1, "title": "...", "publisher": {...}, "authors": [...]}, ...]
```

## Relation Types

### ForeignKey (select_related)

Use `select_related()` for efficient single-query JOINs. Returns a nested **dict**.

```python
Book.objects.select_related("publisher").values_nested()
# {"id": 1, "title": "...", "publisher": {"id": 1, "name": "...", "country": "..."}}
```

### ForeignKey (prefetch_related)

Also supported but less efficient (2 queries instead of 1). Returns a nested **dict**.

```python
Book.objects.prefetch_related("publisher").values_nested()
# {"id": 1, "title": "...", "publisher": {"id": 1, "name": "...", "country": "..."}}
```

### ManyToMany

Use `prefetch_related()`. Returns a **list of dicts**.

```python
Book.objects.prefetch_related("authors").values_nested()
# {"id": 1, "title": "...", "authors": [{"id": 1, "name": "..."}, {"id": 2, "name": "..."}]}
```

### Reverse ForeignKey

Use `prefetch_related()`. Returns a **list of dicts**.

```python
Book.objects.prefetch_related("chapters").values_nested()
# {"id": 1, "title": "...", "chapters": [{"id": 1, "title": "Chapter 1"}, ...]}
```

## Controlling Fields

### Main Model Fields

Use `.only()` to select specific fields:

```python
Book.objects.only("title", "price").values_nested()
# {"id": 1, "title": "...", "price": 29.99}
```

### Related Model Fields (select_related)

Use `.only()` with double-underscore syntax:

```python
Book.objects.only("title", "publisher__name").select_related("publisher").values_nested()
# {"id": 1, "title": "...", "publisher": {"id": 1, "name": "..."}}
```

### Related Model Fields (prefetch_related)

Use `Prefetch` objects with `.only()` on the inner queryset:

```python
from django.db.models import Prefetch

Book.objects.only("title").prefetch_related(
    Prefetch("authors", queryset=Author.objects.only("name"))
).values_nested()
# {"id": 1, "title": "...", "authors": [{"id": 1, "name": "..."}]}
```

## Query Efficiency

| Relation Type | Method | Queries |
|---------------|--------|---------|
| ForeignKey | `select_related()` | 1 (JOIN) |
| ForeignKey | `prefetch_related()` | 2 |
| ManyToMany | `prefetch_related()` | 2 |
| Reverse FK | `prefetch_related()` | 2 |

### Combined Example

```python
# 1 (JOIN for publisher) + 1 (prefetch authors) + 1 (prefetch tags) = 3 queries
Book.objects.select_related("publisher").prefetch_related("authors", "tags").values_nested()
```

## Prefetch Objects

Full support for Django's `Prefetch` objects:

```python
from django.db.models import Prefetch

# Filter prefetched data
Book.objects.prefetch_related(
    Prefetch("chapters", queryset=Chapter.objects.filter(page_count__gt=30))
).values_nested()

# Custom to_attr
Book.objects.prefetch_related(
    Prefetch("chapters", queryset=Chapter.objects.filter(number=1), to_attr="first_chapter")
).values_nested()
# {"id": 1, "title": "...", "first_chapter": [{"id": 1, "title": "Introduction"}]}
```
