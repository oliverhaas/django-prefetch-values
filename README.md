# Django Nested Values

[![PyPI version](https://img.shields.io/pypi/v/django-nested-values.svg)](https://pypi.org/project/django-nested-values/)
[![CI](https://github.com/oliverhaas/django-nested-values/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverhaas/django-nested-values/actions/workflows/ci.yml)

An experimental package that adds `.values_nested()` to Django querysets, returning nested dictionaries with prefetched relations included.

## Quick Example

```python
Book.objects.prefetch_related("authors").values_nested("title", "authors")
# [{"title": "...", "authors": [{"id": 1, "name": "..."}, ...]}]
```

## Documentation

See the [full documentation](https://oliverhaas.github.io/django-nested-values/) for installation, usage, and API reference.

## Supported Versions

|         | Python 3.13 | Python 3.14 |
|---------|:-----------:|:-----------:|
| Django 5.2 | ✓ | ✓ |
| Django 6.0 | ✓ | ✓ |

## License

MIT
