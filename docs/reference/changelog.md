# Changelog

## [Unreleased]

## [0.4.0]

- **Breaking**: `values_nested()` no longer takes arguments
- **New**: Use `.only()` to control which fields are returned
- **New**: Use `.select_related()` for ForeignKey relations (efficient JOINs)
- **New**: Use `.prefetch_related()` for ManyToMany and reverse ForeignKey relations

Migration from 0.3.0:
```python
# Before (0.3.0)
Book.objects.prefetch_related("authors").values_nested("title", "authors")

# After (0.4.0)
Book.objects.only("title").prefetch_related("authors").values_nested()
```

## [0.3.0]

- **Breaking**: Package renamed from `django-prefetch-values` to `django-nested-values`
- **Breaking**: Class renamed from `PrefetchValuesQuerySet` to `NestedValuesQuerySet`
- **Breaking**: Method renamed from `.values()` to `.values_nested()` for API clarity

## [0.1.0]

- Initial release
- `NestedValuesQuerySet` enabling `.prefetch_related().values_nested()` in Django ORM
- Support for ManyToMany, reverse ForeignKey, reverse ManyToMany relations
- Support for Django's `Prefetch` object with custom querysets and `to_attr`
- Support for nested prefetches
