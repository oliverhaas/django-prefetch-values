# Django Prefetch Values

An experimental package exploring how to combine `.prefetch_related()` with `.values()` in Django ORM, mainly so this can be used in some test cases to evaluate whether this feature is worth the effort.

This explores a solution to [Django ticket #26565](https://code.djangoproject.com/ticket/26565).

## Quick Example

```python
from django_prefetch_values import PrefetchValuesQuerySet

class Book(models.Model):
    title = models.CharField(max_length=200)
    authors = models.ManyToManyField("Author")

    objects = PrefetchValuesQuerySet.as_manager()

# Returns dicts with nested prefetched data
books = Book.objects.prefetch_related("authors").values()
# [{"id": 1, "title": "Book 1", "authors": [{"id": 1, "name": "Author 1"}, ...]}, ...]
```

## Requirements

- Python 3.13+
- Django 5.2+
