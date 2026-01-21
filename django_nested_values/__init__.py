"""Django Nested Values - Enable .prefetch_related().values_nested() in Django ORM."""

from django_nested_values.queryset import (
    AttrDict,
    NestedValuesQuerySet,
    NestedValuesQuerySetMixin,
)

__all__ = ["AttrDict", "NestedValuesQuerySet", "NestedValuesQuerySetMixin"]
__version__ = "0.5.2"
