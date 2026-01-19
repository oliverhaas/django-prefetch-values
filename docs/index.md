# Django Nested Values

An experimental package that adds `.values_nested()` to Django querysets, returning nested dictionaries with prefetched relations included.

## Quick Example

```python
from django_nested_values import NestedValuesQuerySet

class Book(models.Model):
    title = models.CharField(max_length=200)
    authors = models.ManyToManyField("Author")

    objects = NestedValuesQuerySet.as_manager()

# Returns dicts with nested prefetched data
books = Book.objects.prefetch_related("authors").values_nested()
# [{"id": 1, "title": "Book 1", "authors": [{"id": 1, "name": "Author 1"}, ...]}, ...]
```

## Requirements

- Python 3.13+
- Django 5.2+
