"""Django Nested Values - Enable .prefetch_related().values_nested() in Django ORM."""

from django_nested_values.queryset import (
    NestedObject,
    NestedValuesQuerySet,
    NestedValuesQuerySetMixin,
    RelatedList,
)

__all__ = ["NestedObject", "NestedValuesQuerySet", "NestedValuesQuerySetMixin", "RelatedList"]
__version__ = "0.5.2"
