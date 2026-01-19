# Django Nested Values

An experimental package that adds `.values_nested()` to Django querysets, returning nested dictionaries with related objects included.

## Quick Example

```python
from django_nested_values import NestedValuesQuerySet

class Book(models.Model):
    title = models.CharField(max_length=200)
    publisher = models.ForeignKey(Publisher, on_delete=models.CASCADE)
    authors = models.ManyToManyField("Author")

    objects = NestedValuesQuerySet.as_manager()

# Returns dicts with nested related data
books = (
    Book.objects
    .only("title")
    .select_related("publisher")
    .prefetch_related("authors")
    .values_nested()
)
# [{"id": 1, "title": "...", "publisher": {"id": 1, "name": "..."}, "authors": [...]}, ...]
```

## Key Features

- **Familiar API**: Uses standard Django patterns (`only()`, `select_related()`, `prefetch_related()`)
- **ForeignKey via JOIN**: `select_related()` uses efficient single-query JOINs
- **M2M/Reverse FK via prefetch**: `prefetch_related()` for multi-valued relations
- **No model instantiation**: Returns dicts directly from the database

## Requirements

- Python 3.13+
- Django 5.2+
