# Changelog

## [Unreleased]

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
