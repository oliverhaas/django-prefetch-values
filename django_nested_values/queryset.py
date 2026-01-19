"""Custom QuerySet that adds .values_nested() for nested dictionaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast

from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.prefetch import GenericPrefetch
from django.core.exceptions import FieldDoesNotExist
from django.db.models import ForeignKey, ManyToManyField, ManyToManyRel, ManyToOneRel, Model, Prefetch, QuerySet
from django.db.models.query import BaseIterable

# TypeVar for the model type, used for generic typing with django-stubs
_ModelT_co = TypeVar("_ModelT_co", bound=Model, covariant=True)

if TYPE_CHECKING:
    from collections.abc import Iterator

    # For type checking, pretend the mixin inherits from QuerySet
    # This allows type checkers to see QuerySet methods on the mixin
    class _MixinBase(QuerySet[_ModelT_co, _ModelT_co]):
        pass
else:
    # At runtime, use Generic to allow subscripting like NestedValuesQuerySetMixin[Book]
    _MixinBase = Generic


class NestedValuesIterable(BaseIterable):
    """Iterable that yields nested dictionaries for QuerySet.values_nested().

    This follows Django's pattern of using iterable classes (like ValuesIterable)
    to control how queryset iteration yields results.
    """

    if TYPE_CHECKING:
        # The queryset is expected to be a NestedValuesQuerySetMixin
        queryset: NestedValuesQuerySetMixin[Any]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the queryset, yielding nested dictionaries."""
        queryset = self.queryset

        # Determine which fields to fetch based on .only() / .defer()
        main_fields = queryset._get_main_fields()
        select_related_fields = queryset._get_select_related_fields()
        prefetch_lookups = queryset._prefetch_related_lookups  # type: ignore[attr-defined]

        pk_name = queryset.model._meta.pk.name

        # Build fields for main query (include pk, main fields, and select_related fields)
        query_fields = queryset._build_query_fields(main_fields, select_related_fields, pk_name)

        # Execute main query with select_related joins
        main_qs = queryset._build_main_queryset()
        main_results = list(main_qs.values(*query_fields))

        if not main_results:
            return

        # Get PKs for prefetch queries
        pk_values = [r[pk_name] for r in main_results]

        # Fetch prefetched relations (pass main_results to avoid extra queries for FK ids)
        prefetched_data = queryset._fetch_all_prefetched(prefetch_lookups, pk_values, main_results)

        # Build and yield final results
        yield from queryset._build_results(
            main_results,
            main_fields,
            select_related_fields,
            prefetched_data,
            pk_name,
        )


class NestedValuesQuerySetMixin(_MixinBase[_ModelT_co]):
    """Mixin that adds .values_nested() to any QuerySet.

    Use this mixin to add values_nested() to your custom QuerySet classes:

        class MyQuerySet(NestedValuesQuerySetMixin, QuerySet):
            def my_custom_method(self):
                ...

        class Book(models.Model):
            objects = MyQuerySet.as_manager()

    Or use the pre-built NestedValuesQuerySet if you don't need a custom QuerySet.

    Type hints: After calling values_nested(), the queryset yields dict[str, Any]
    when iterated, similar to Django's values() method.
    """

    def values_nested(self) -> QuerySet[_ModelT_co, dict[str, Any]]:
        """Return nested dictionaries with related objects included.

        Takes no arguments. Use standard Django methods to control output:
        - .only() to select which fields to include
        - .select_related() for ForeignKey relations (single dict)
        - .prefetch_related() for ManyToMany/reverse FK relations (list of dicts)

        Returns:
            A QuerySet that yields dict[str, Any] when iterated, with nested
            dictionaries for related objects.
        """
        clone: Self = self._clone()  # type: ignore[attr-defined]  # _clone is from QuerySet
        clone._iterable_class = NestedValuesIterable
        return clone  # Return type changes from Self to QuerySet[..., dict]

    def _get_main_fields(self) -> list[str]:
        """Get the fields to fetch for the main model based on .only() / .defer().

        Uses attname for FK fields (e.g., publisher_id instead of publisher) to match
        Django's .values() behavior.
        """
        # Check if .only() or .defer() was used
        deferred = self.query.deferred_loading
        deferred_fields, is_defer = deferred

        if not deferred_fields:
            # No .only() or .defer() - return all concrete fields
            # Use attname for FK fields (e.g., publisher_id instead of publisher)
            return [f.attname for f in self.model._meta.concrete_fields]

        if is_defer:
            # .defer() was used - return all fields except deferred ones
            all_fields = {f.attname for f in self.model._meta.concrete_fields}
            return list(all_fields - deferred_fields)
        # .only() was used - return only those fields
        # Filter out double-underscore fields (those are for relations, not main model)
        # Django always includes pk even if not explicitly specified
        pk_name = self.model._meta.pk.name
        fields = [f for f in deferred_fields if "__" not in f]
        if pk_name not in fields:
            fields.insert(0, pk_name)
        return fields

    def _get_select_related_fields(self) -> dict[str, list[str]]:
        """Get select_related relations and their fields.

        Returns a dict mapping relation path to list of fields to fetch.
        Empty list means all fields.
        Nested relations are flattened: {'book': {'publisher': {}}} -> {'book': [], 'book__publisher': []}
        """
        select_related = self.query.select_related

        if not select_related:
            return {}

        if select_related is True:
            # select_related() with no args - Django auto-follows all FK/OneToOne
            # We'll handle this by getting all FK fields
            result = {}
            for field in self.model._meta.concrete_fields:
                if isinstance(field, ForeignKey):
                    result[field.name] = []
            return result

        # select_related is a dict of {relation_name: nested_select_related}
        # Flatten nested structure to paths like "book__publisher"
        result = {}
        self._flatten_select_related(select_related, "", result)
        return result

    def _flatten_select_related(
        self,
        select_related: dict,
        prefix: str,
        result: dict[str, list[str]],
    ) -> None:
        """Recursively flatten nested select_related structure to path-based dict."""
        for relation_name, nested in select_related.items():
            full_path = f"{prefix}{relation_name}" if prefix else relation_name
            # Get the fields for this relation based on .only()
            relation_fields = self._get_relation_fields_from_only(full_path)
            result[full_path] = relation_fields

            # Recurse for nested select_related
            if nested:
                self._flatten_select_related(nested, f"{full_path}__", result)

    def _get_relation_fields_from_only(self, relation_name: str) -> list[str]:
        """Get the fields to fetch for a relation based on .only() with double-underscore."""
        deferred = self.query.deferred_loading
        deferred_fields, is_defer = deferred

        if not deferred_fields or is_defer:
            # No .only() or using .defer() - return empty (means all fields)
            return []

        # Check for fields like "publisher__name" in only()
        prefix = f"{relation_name}__"
        return [field[len(prefix) :] for field in deferred_fields if field.startswith(prefix)]

    def _build_query_fields(
        self,
        main_fields: list[str],
        select_related_fields: dict[str, list[str]],
        pk_name: str,
    ) -> list[str]:
        """Build the list of fields for the main .values() query."""
        # Start with pk (always needed for prefetch lookups)
        fields = [pk_name] if pk_name not in main_fields else []

        # Get FK attnames that are being select_related'd at top level
        # (we don't want both publisher_id and publisher.*)
        fk_attnames_to_skip = set()
        for relation_path in select_related_fields:
            # Only skip for top-level relations (no __ in path)
            if "__" not in relation_path:
                try:
                    rel_field = self.model._meta.get_field(relation_path)
                    if isinstance(rel_field, ForeignKey):
                        fk_attnames_to_skip.add(rel_field.attname)  # e.g., publisher_id
                except FieldDoesNotExist:
                    pass

        # Add main model fields (excluding FKs that have select_related)
        for field in main_fields:
            if field not in fk_attnames_to_skip and "__" not in field and field not in fields:
                fields.append(field)

        # Add select_related fields with double-underscore notation
        for relation_path, relation_fields in select_related_fields.items():
            # Get the related model by traversing the path
            related_model = self._get_related_model_for_path(relation_path)
            if related_model is None:
                continue

            related_pk = related_model._meta.pk.name

            if relation_fields:
                # Specific fields requested via only()
                # Always include pk for the relation
                if related_pk not in relation_fields:
                    fields.append(f"{relation_path}__{related_pk}")
                fields.extend(f"{relation_path}__{rf}" for rf in relation_fields)
            else:
                # All fields from the related model
                fields.extend(f"{relation_path}__{rf.name}" for rf in related_model._meta.concrete_fields)

        return fields

    def _get_related_model_for_path(self, path: str) -> type[Model] | None:
        """Get the related model for a relation path like 'book__publisher'."""
        parts = path.split("__")
        current_model = self.model

        for part in parts:
            try:
                rel_field = current_model._meta.get_field(part)
                if isinstance(rel_field, ForeignKey):
                    current_model = rel_field.related_model
                else:
                    return None
            except FieldDoesNotExist:
                return None

        return current_model

    def _build_main_queryset(self) -> QuerySet:
        """Build a fresh queryset for the main query."""
        main_qs = self.model._default_manager.using(self.db).all()
        main_qs.query = self.query.chain()
        main_qs.query.values_select = ()
        return main_qs

    def _fetch_all_prefetched(
        self,
        prefetch_lookups: tuple,
        parent_pks: list[Any],
        main_results: list[dict],
    ) -> dict[str, dict[Any, list[dict] | dict | None]]:
        """Fetch all prefetched relations."""
        result: dict[str, dict[Any, list[dict] | dict | None]] = {}

        # Group lookups by top-level relation name
        lookup_map = self._group_prefetch_lookups(prefetch_lookups)

        for attr_name, lookup_info in lookup_map.items():
            lookup = lookup_info["lookup"]
            nested = lookup_info["nested"]

            if lookup_info.get("is_generic_fk"):
                # Handle GenericForeignKey via GenericPrefetch
                result[attr_name] = self._fetch_generic_fk_values(  # type: ignore[assignment]
                    lookup,
                    parent_pks,
                    main_results,
                )
            else:
                result[attr_name] = self._fetch_relation_values(
                    attr_name,
                    lookup,
                    nested,
                    parent_pks,
                    main_results,
                )

        return result

    def _group_prefetch_lookups(self, prefetch_lookups: tuple) -> dict[str, dict]:
        """Group prefetch lookups by their top-level attribute name."""
        result: dict[str, dict] = {}

        for lookup in prefetch_lookups:
            if isinstance(lookup, GenericPrefetch):
                # GenericPrefetch for GenericForeignKey
                attr_name = lookup.to_attr or lookup.prefetch_to
                result[attr_name] = {"lookup": lookup, "nested": [], "is_generic_fk": True}
            elif isinstance(lookup, Prefetch):
                # Get the attribute name (to_attr or top-level relation)
                to_attr, _ = lookup.get_current_to_attr(0)
                attr_name = to_attr or lookup.prefetch_to.split("__")[0]
                relation_path = lookup.prefetch_through if to_attr else lookup.prefetch_to

                # Track nested relations
                if attr_name not in result:
                    result[attr_name] = {"lookup": lookup, "nested": []}

                # Check for nested parts (e.g., "books__chapters" -> "chapters" is nested under "books")
                if "__" in relation_path:
                    parts = relation_path.split("__", 1)
                    if parts[0] == attr_name or attr_name == parts[0]:
                        result[attr_name]["nested"].append(parts[1])
            else:
                # String lookup
                attr_name = lookup.split("__")[0]
                relation_path = lookup

                # Track nested relations
                if attr_name not in result:
                    result[attr_name] = {"lookup": lookup, "nested": []}

                # Check for nested parts
                if "__" in relation_path:
                    parts = relation_path.split("__", 1)
                    if parts[0] == attr_name:
                        result[attr_name]["nested"].append(parts[1])

        return result

    def _fetch_relation_values(
        self,
        attr_name: str,
        lookup: str | Prefetch,
        nested_relations: list[str],
        parent_pks: list[Any],
        main_results: list[dict],
    ) -> dict[Any, list[dict] | dict | None]:
        """Fetch a related model's data and group by parent PK."""
        # Get the actual relation name (may differ from attr_name if to_attr is used)
        if isinstance(lookup, Prefetch):
            relation_name = lookup.prefetch_through.split("__")[0]
        else:
            relation_name = lookup.split("__")[0]

        # Find the relation field
        try:
            field = self.model._meta.get_field(relation_name)
        except FieldDoesNotExist:
            return {}

        # Get the custom queryset if using Prefetch
        custom_qs = None
        if isinstance(lookup, Prefetch) and lookup.queryset is not None:
            custom_qs = lookup.queryset

        # Fetch based on relation type
        if isinstance(field, ManyToManyField):
            return self._fetch_m2m_values(field, nested_relations, parent_pks, custom_qs, main_results)  # type: ignore[return-value]
        if isinstance(field, ManyToOneRel):
            return self._fetch_reverse_fk_values(field, nested_relations, parent_pks, custom_qs, main_results)  # type: ignore[return-value]
        if isinstance(field, ForeignKey):
            return self._fetch_fk_values(field, nested_relations, parent_pks, main_results, custom_qs)  # type: ignore[return-value]
        if isinstance(field, ManyToManyRel):
            return self._fetch_reverse_m2m_values(field, nested_relations, parent_pks, custom_qs, main_results)  # type: ignore[return-value]
        if isinstance(field, GenericRelation):
            return self._fetch_generic_relation_values(field, nested_relations, parent_pks, custom_qs, main_results)  # type: ignore[return-value]

        return {}

    def _get_fields_for_relation(self, related_model: type[Any], custom_qs: QuerySet | None) -> list[str]:
        """Get the fields to fetch for a related model."""
        if custom_qs is not None:
            # Check if the custom queryset has .only()
            deferred = custom_qs.query.deferred_loading
            deferred_fields, is_defer = deferred

            if deferred_fields and not is_defer:
                # .only() was used
                return list(deferred_fields)

        # Return all concrete fields
        return [f.name for f in related_model._meta.concrete_fields]

    def _extract_fk_data_from_main_results(
        self,
        main_results: list[dict],
        relation_name: str,
        related_model: type[Model],
    ) -> dict[Any, dict]:
        """Extract FK relation data that was already fetched via select_related.

        When a FK is fetched via select_related, the data is already in main_results
        with keys like "relation__field". This extracts that data into a dict
        keyed by the related object's PK.
        """
        related_pk_name = related_model._meta.pk.name
        prefix = f"{relation_name}__"

        result: dict[Any, dict] = {}

        for row in main_results:
            # Extract all fields with the relation prefix
            related_dict: dict[str, Any] = {}
            for key, value in row.items():
                if key.startswith(prefix):
                    field_name = key[len(prefix) :]
                    # Skip nested relations (contain another __)
                    if "__" not in field_name:
                        related_dict[field_name] = value

            # Get the related object's PK
            related_pk = related_dict.get(related_pk_name)
            if related_pk is not None and related_pk not in result:
                result[related_pk] = related_dict

        return result

    def _fetch_m2m_values(
        self,
        field: ManyToManyField,
        nested_relations: list[str],
        parent_pks: list[Any],
        custom_qs: QuerySet | None,
        main_results: list[dict] | None = None,
    ) -> dict[Any, list[dict]]:
        """Fetch ManyToMany relation data."""
        related_model = field.related_model
        reverse_accessor = field.related_query_name()

        fetch_fields = self._get_fields_for_relation(related_model, custom_qs)
        related_pk_name = related_model._meta.pk.name
        if related_pk_name not in fetch_fields:
            fetch_fields = [related_pk_name, *fetch_fields]

        # Build queryset
        if custom_qs is not None:
            related_qs = custom_qs.filter(**{f"{reverse_accessor}__in": parent_pks})
        else:
            related_qs = related_model._default_manager.filter(**{f"{reverse_accessor}__in": parent_pks})

        # Add parent PK to values for grouping
        fetch_fields_with_source = [*fetch_fields, f"{reverse_accessor}__pk"]
        raw_data = list(related_qs.values(*fetch_fields_with_source))

        # Group by parent PK
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        related_data: dict[Any, dict] = {}

        for row in raw_data:
            source_pk = row.pop(f"{reverse_accessor}__pk")
            related_pk = row[related_pk_name]
            result[source_pk].append(dict(row))

            if nested_relations and related_pk not in related_data:
                related_data[related_pk] = dict(row)

        # Handle nested relations
        if nested_relations and related_data:
            # Parent path is the relation name + "__" for nested relations
            parent_path = f"{field.name}__"
            self._add_nested_relations(
                related_model,
                related_data,
                nested_relations,
                list(related_data.keys()),
                main_results,
                parent_path,
            )
            # Update results with nested data
            for source_pk, items in result.items():
                for item in items:
                    pk_val = item.get(related_pk_name)
                    if pk_val and pk_val in related_data:
                        for key, val in related_data[pk_val].items():
                            if key not in item:
                                item[key] = val

        return result

    def _fetch_reverse_fk_values(
        self,
        field: ManyToOneRel,
        nested_relations: list[str],
        parent_pks: list[Any],
        custom_qs: QuerySet | None,
        main_results: list[dict] | None = None,
    ) -> dict[Any, list[dict]]:
        """Fetch reverse ForeignKey relation data."""
        related_model = field.related_model
        fk_field_name = field.field.name

        fetch_fields = self._get_fields_for_relation(related_model, custom_qs)
        related_pk_name = related_model._meta.pk.name
        if related_pk_name not in fetch_fields:
            fetch_fields = [related_pk_name, *fetch_fields]
        if fk_field_name not in fetch_fields:
            fetch_fields = [*fetch_fields, fk_field_name]

        # Build queryset
        if custom_qs is not None:
            related_qs = custom_qs.filter(**{f"{fk_field_name}__in": parent_pks})
        else:
            related_qs = related_model._default_manager.filter(**{f"{fk_field_name}__in": parent_pks})

        related_data = list(related_qs.values(*fetch_fields))

        # Handle nested relations
        if nested_relations and related_data:
            related_pks = [r[related_pk_name] for r in related_data]
            nested_dict = {r[related_pk_name]: dict(r) for r in related_data}
            # Parent path is the relation accessor name + "__"
            parent_path = f"{field.get_accessor_name()}__"
            self._add_nested_relations(
                related_model,
                nested_dict,
                nested_relations,
                related_pks,
                main_results,
                parent_path,
            )
            related_data = list(nested_dict.values())

        # Group by parent PK
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row[fk_field_name]
            row_dict = dict(row)
            row_dict.pop(fk_field_name, None)
            result[parent_pk].append(row_dict)

        return result

    def _fetch_fk_values(
        self,
        field: ForeignKey,
        nested_relations: list[str],
        parent_pks: list[Any],
        main_results: list[dict],
        custom_qs: QuerySet | None,
    ) -> dict[Any, dict | None]:
        """Fetch ForeignKey relation data (when using prefetch_related, not select_related)."""
        related_model = field.related_model
        fk_attname = field.attname  # e.g., publisher_id
        relation_name = field.name  # e.g., publisher
        related_pk_name = related_model._meta.pk.name

        # Get FK values from main_results (avoids extra query)
        pk_name = self.model._meta.pk.name
        fk_data = {}
        for r in main_results:
            parent_pk = r[pk_name]
            # Try direct FK attname first (when not using select_related)
            fk_value = r.get(fk_attname)
            # Fallback to select_related data (when FK attname was skipped)
            if fk_value is None:
                fk_value = r.get(f"{relation_name}__{related_pk_name}")
            fk_data[parent_pk] = fk_value

        fk_values = list({v for v in fk_data.values() if v is not None})
        if not fk_values:
            return dict.fromkeys(parent_pks)

        # Check if this relation was already fetched via select_related
        # If so, we can extract the data from main_results instead of querying again
        select_related_fields = self._get_select_related_fields()
        if relation_name in select_related_fields and custom_qs is None:
            # Data already in main_results from JOIN - extract it directly
            related_data = self._extract_fk_data_from_main_results(
                main_results,
                relation_name,
                related_model,
            )
        else:
            # Not select_related or has custom queryset - need to query
            fetch_fields = self._get_fields_for_relation(related_model, custom_qs)
            if related_pk_name not in fetch_fields:
                fetch_fields = [related_pk_name, *fetch_fields]

            if custom_qs is not None:
                related_qs = custom_qs.filter(pk__in=fk_values)
            else:
                related_qs = related_model._default_manager.filter(pk__in=fk_values)

            related_data = {r[related_pk_name]: dict(r) for r in related_qs.values(*fetch_fields)}

        # Handle nested relations
        if nested_relations:
            # Parent path is the relation name + "__" for nested relations
            parent_path = f"{relation_name}__"
            self._add_nested_relations(
                related_model,
                related_data,
                nested_relations,
                fk_values,
                main_results,
                parent_path,
            )

        # Map to parent PKs
        result: dict[Any, dict | None] = {}
        for parent_pk in parent_pks:
            fk_value = fk_data.get(parent_pk)
            if fk_value is not None and fk_value in related_data:
                result[parent_pk] = dict(related_data[fk_value])
            else:
                result[parent_pk] = None

        return result

    def _fetch_reverse_m2m_values(
        self,
        field: ManyToManyRel,
        nested_relations: list[str],
        parent_pks: list[Any],
        custom_qs: QuerySet | None,
        main_results: list[dict] | None = None,
    ) -> dict[Any, list[dict]]:
        """Fetch reverse ManyToMany relation data."""
        related_model = field.related_model
        through_model = field.through
        m2m_field = field.field

        source_col = m2m_field.m2m_reverse_name()
        target_col = m2m_field.m2m_column_name()

        through_qs = through_model.objects.filter(**{f"{source_col}__in": parent_pks})  # type: ignore[union-attr]
        through_data = list(through_qs.values(source_col, target_col))

        if not through_data:
            return {pk: [] for pk in parent_pks}

        related_pks = [t[target_col] for t in through_data]

        fetch_fields = self._get_fields_for_relation(related_model, custom_qs)
        related_pk_name = related_model._meta.pk.name
        if related_pk_name not in fetch_fields:
            fetch_fields = [related_pk_name, *fetch_fields]

        # Build queryset
        if custom_qs is not None:
            related_qs = custom_qs.filter(pk__in=related_pks)
        else:
            related_qs = related_model._default_manager.filter(pk__in=related_pks)

        related_data = {r[related_pk_name]: dict(r) for r in related_qs.values(*fetch_fields)}

        # Handle nested relations
        if nested_relations:
            # Parent path is the relation accessor name + "__"
            parent_path = f"{field.get_accessor_name()}__"
            self._add_nested_relations(
                related_model,
                related_data,
                nested_relations,
                list(related_data.keys()),
                main_results,
                parent_path,
            )

        # Group by parent PK
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for through_row in through_data:
            parent_pk = through_row[source_col]
            related_pk = through_row[target_col]
            if related_pk in related_data:
                result[parent_pk].append(dict(related_data[related_pk]))

        return result

    def _fetch_generic_relation_values(
        self,
        field: GenericRelation,
        nested_relations: list[str],
        parent_pks: list[Any],
        custom_qs: QuerySet | None,
        main_results: list[dict] | None = None,
    ) -> dict[Any, list[dict]]:
        """Fetch GenericRelation data (content types framework)."""
        related_model = field.related_model
        ct_field_name = field.content_type_field_name  # Usually "content_type"
        obj_id_field_name = field.object_id_field_name  # Usually "object_id"

        # Get the ContentType for the parent model
        parent_ct = ContentType.objects.get_for_model(self.model)

        fetch_fields = self._get_fields_for_relation(related_model, custom_qs)
        related_pk_name = related_model._meta.pk.name
        if related_pk_name not in fetch_fields:
            fetch_fields = [related_pk_name, *fetch_fields]
        # Ensure object_id is in fetch_fields for grouping
        if obj_id_field_name not in fetch_fields:
            fetch_fields = [*fetch_fields, obj_id_field_name]

        # Build queryset - filter by content_type and object_id
        if custom_qs is not None:
            related_qs = custom_qs.filter(
                **{ct_field_name: parent_ct, f"{obj_id_field_name}__in": parent_pks},
            )
        else:
            related_qs = related_model._default_manager.filter(
                **{ct_field_name: parent_ct, f"{obj_id_field_name}__in": parent_pks},
            )

        related_data = list(related_qs.values(*fetch_fields))

        # Handle nested relations
        if nested_relations and related_data:
            related_pks = [r[related_pk_name] for r in related_data]
            nested_dict = {r[related_pk_name]: dict(r) for r in related_data}
            # Parent path is the relation name + "__"
            parent_path = f"{field.name}__"
            self._add_nested_relations(
                related_model,
                nested_dict,
                nested_relations,
                related_pks,
                main_results,
                parent_path,
            )
            related_data = list(nested_dict.values())

        # Group by parent PK (object_id)
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row[obj_id_field_name]
            row_dict = dict(row)
            # Remove the object_id field from output (it's the parent PK, not useful in nested output)
            row_dict.pop(obj_id_field_name, None)
            # Also remove content_type_id if present (internal FK, not useful)
            row_dict.pop(f"{ct_field_name}_id", None)
            result[parent_pk].append(row_dict)

        return result

    def _fetch_generic_fk_values(
        self,
        lookup: GenericPrefetch,
        parent_pks: list[Any],
        main_results: list[dict],
    ) -> dict[Any, dict | None]:
        """Fetch GenericForeignKey data using GenericPrefetch.

        GenericForeignKey can point to different model types. GenericPrefetch
        provides a list of querysets, one for each possible content type.
        """
        gfk_attr = lookup.prefetch_to  # e.g., "content_object"

        # Find the GenericForeignKey descriptor on the model to get field names
        gfk_descriptor = getattr(self.model, gfk_attr, None)
        if not isinstance(gfk_descriptor, GenericForeignKey):
            return dict.fromkeys(parent_pks)

        ct_field = gfk_descriptor.ct_field  # e.g., "content_type"
        fk_field = gfk_descriptor.fk_field  # e.g., "object_id"
        ct_attname = f"{ct_field}_id"  # e.g., "content_type_id"

        pk_name = self.model._meta.pk.name

        # Build mapping: parent_pk -> (content_type_id, object_id)
        parent_gfk_info: dict[Any, tuple[Any, Any]] = {}
        for row in main_results:
            parent_pk = row[pk_name]
            ct_id = row.get(ct_attname)
            obj_id = row.get(fk_field)
            parent_gfk_info[parent_pk] = (ct_id, obj_id)

        # Group parent PKs by content_type_id
        ct_to_parents: dict[Any, list[tuple[Any, Any]]] = {}  # ct_id -> [(parent_pk, object_id), ...]
        for parent_pk, (ct_id, obj_id) in parent_gfk_info.items():
            if ct_id is not None and obj_id is not None:
                if ct_id not in ct_to_parents:
                    ct_to_parents[ct_id] = []
                ct_to_parents[ct_id].append((parent_pk, obj_id))

        # Build mapping: content_type_id -> queryset from GenericPrefetch
        ct_to_queryset: dict[int, QuerySet] = {}
        for qs in lookup.querysets:  # type: ignore[attr-defined]
            ct = ContentType.objects.get_for_model(qs.model)
            ct_to_queryset[ct.id] = qs

        # Fetch objects for each content type
        result: dict[Any, dict | None] = dict.fromkeys(parent_pks)

        for ct_id, parent_obj_pairs in ct_to_parents.items():
            if ct_id not in ct_to_queryset:
                # Content type not in the provided querysets - skip
                continue

            qs = ct_to_queryset[ct_id]
            related_model = qs.model
            related_pk_name = related_model._meta.pk.name

            # Get object IDs to fetch
            object_ids = [obj_id for _, obj_id in parent_obj_pairs]

            # Get fields to fetch
            fetch_fields = self._get_fields_for_relation(related_model, qs)
            if related_pk_name not in fetch_fields:
                fetch_fields = [related_pk_name, *fetch_fields]

            # Fetch the related objects
            related_qs = qs.filter(pk__in=object_ids)
            related_data = {r[related_pk_name]: dict(r) for r in related_qs.values(*fetch_fields)}

            # Handle nested prefetches on the GenericPrefetch queryset
            if qs._prefetch_related_lookups:  # type: ignore[attr-defined]
                # The queryset has its own prefetch_related - we need to process those
                nested_pks = list(related_data.keys())
                if nested_pks:
                    nested_prefetched = self._fetch_prefetched_for_related(
                        related_model,
                        qs._prefetch_related_lookups,  # type: ignore[attr-defined]
                        nested_pks,
                        list(related_data.values()),
                    )
                    # Merge nested data into related_data
                    for pk_val, row_data in related_data.items():
                        for attr, data_by_pk in nested_prefetched.items():
                            row_data[attr] = data_by_pk.get(pk_val, [])

            # Map back to parent PKs
            for parent_pk, obj_id in parent_obj_pairs:
                if obj_id in related_data:
                    result[parent_pk] = dict(related_data[obj_id])

        return result

    def _fetch_prefetched_for_related(
        self,
        model: type[Model],
        prefetch_lookups: tuple,
        parent_pks: list[Any],
        main_results: list[dict],
    ) -> dict[str, dict[Any, list[dict] | dict | None]]:
        """Fetch prefetched relations for a related model (used by GenericForeignKey)."""
        # Create a temporary mixin instance for the related model
        temp_qs = NestedValuesQuerySetMixin.__new__(NestedValuesQuerySetMixin)
        temp_qs.model = model
        temp_qs.db = self.db
        temp_qs.query = model._default_manager.all().query

        return temp_qs._fetch_all_prefetched(prefetch_lookups, parent_pks, main_results)

    def _build_results(
        self,
        main_results: list[dict],
        main_fields: list[str],
        select_related_fields: dict[str, list[str]],
        prefetched_data: dict[str, dict[Any, list[dict] | dict | None]],
        pk_name: str,
    ) -> Iterator[dict[str, Any]]:
        """Build the final nested dictionaries."""
        # Get only top-level select_related (no __ in path)
        top_level_select_related = {k: v for k, v in select_related_fields.items() if "__" not in k}

        for row in main_results:
            row_dict: dict[str, Any] = {}
            pk = row[pk_name]

            # Add pk if it was in main_fields
            if pk_name in main_fields:
                row_dict[pk_name] = pk

            # Add main model fields
            for field in main_fields:
                if field in row and field not in top_level_select_related and "__" not in field:
                    row_dict[field] = row[field]

            # Build nested dicts for select_related (only top-level, nested handled recursively)
            for relation_name, relation_fields in top_level_select_related.items():
                nested_dict = self._extract_relation_from_row(
                    row,
                    relation_name,
                    relation_fields,
                    select_related_fields,
                    cast("type[Model]", self.model),
                )
                if nested_dict is not None:
                    row_dict[relation_name] = nested_dict

            # Add prefetched relations
            for attr_name, data_by_pk in prefetched_data.items():
                prefetch_value = data_by_pk.get(pk, [])

                # Check if this attr was already set by select_related
                if attr_name in row_dict:
                    existing = row_dict[attr_name]
                    if isinstance(existing, dict) and isinstance(prefetch_value, dict):
                        # Deep merge prefetched nested data into select_related data
                        # select_related data takes precedence for scalar fields
                        self._deep_merge_dicts(existing, prefetch_value)
                    # If types don't match, keep select_related data (shouldn't happen in normal use)
                else:
                    row_dict[attr_name] = prefetch_value

            yield row_dict

    def _deep_merge_dicts(self, target: dict, source: dict) -> None:
        """Deep merge source dict into target dict.

        For nested dicts, recursively merge. For other types, only add if key doesn't exist.
        This allows prefetch_related to add nested data (like books list) to select_related
        data (like publisher dict) without overwriting existing fields.
        """
        for key, val in source.items():
            if key not in target:
                # Key doesn't exist in target, add it
                target[key] = val
            elif isinstance(target[key], dict) and isinstance(val, dict):
                # Both are dicts, recursively merge
                self._deep_merge_dicts(target[key], val)
            # Otherwise, keep target's value (select_related takes precedence)

    def _extract_relation_from_row(  # noqa: PLR0913
        self,
        row: dict,
        relation_name: str,
        relation_fields: list[str],
        all_select_related: dict[str, list[str]],
        parent_model: type[Model],
        prefix_path: str = "",
    ) -> dict | None:
        """Extract a select_related relation's data from a flat row dict.

        Args:
            row: The flat row dict from values() query
            relation_name: The relation name (e.g., 'book')
            relation_fields: List of fields to include (empty = all fields)
            all_select_related: Full dict of all select_related paths
            parent_model: The model that has this relation
            prefix_path: Path prefix for nested relations (e.g., 'book__' when extracting publisher from book)
        """
        # Build the full prefix for extracting from row
        full_prefix = f"{prefix_path}{relation_name}__"
        nested_dict: dict[str, Any] = {}

        # Get related model and its pk name
        try:
            rel_field = parent_model._meta.get_field(relation_name)
            if not isinstance(rel_field, ForeignKey):
                return None
            related_model = rel_field.related_model
            related_pk_name = related_model._meta.pk.name
        except FieldDoesNotExist:
            return None

        # Build the full path for checking nested select_related
        current_path = f"{prefix_path}{relation_name}" if prefix_path else relation_name

        # Find which nested relations are select_related from this level
        nested_select_related = {}
        nested_prefix = f"{current_path}__"
        for path, fields in all_select_related.items():
            if path.startswith(nested_prefix) and "__" not in path[len(nested_prefix) :]:
                # This is a direct child relation
                child_relation = path[len(nested_prefix) :]
                nested_select_related[child_relation] = fields

        # Extract fields from row
        for key, value in row.items():
            if key.startswith(full_prefix):
                field_name = key[len(full_prefix) :]

                # Skip fields that belong to nested relations (contain __)
                if "__" in field_name:
                    continue

                # Check if this is a FK field that should be a nested dict
                try:
                    field_obj = related_model._meta.get_field(field_name)
                    is_fk = isinstance(field_obj, ForeignKey)
                except FieldDoesNotExist:
                    is_fk = False

                if is_fk:
                    # Always include the _id field (e.g., publisher_id) for FK fields
                    fk_field = field_obj
                    nested_dict[fk_field.attname] = value
                    # Skip adding as relation name - nested dict will be added below if select_related
                    continue

                # Include this field if:
                # - It's the pk field
                # - No specific fields requested (empty list = all fields)
                # - It's in the requested fields list
                if field_name == related_pk_name or not relation_fields or field_name in relation_fields:
                    nested_dict[field_name] = value

        # Recursively extract nested select_related relations
        for child_relation, child_fields in nested_select_related.items():
            child_dict = self._extract_relation_from_row(
                row,
                child_relation,
                child_fields,
                all_select_related,
                related_model,
                full_prefix,
            )
            if child_dict is not None:
                nested_dict[child_relation] = child_dict

        # Check for NULL FK (related pk is None)
        if related_pk_name and nested_dict.get(related_pk_name) is None:
            return None

        return nested_dict if nested_dict else None

    def _extract_nested_fk_from_main_results(
        self,
        main_results: list[dict],
        full_path: str,
        related_model: type[Model],
        parent_pks: list[Any],
    ) -> dict[Any, dict]:
        """Extract nested FK data that was already fetched via select_related.

        When a nested FK is fetched via select_related (e.g., "product__master"),
        the data is already in main_results with keys like "product__master__field".
        This extracts that data into a dict keyed by the related object's PK.

        Args:
            main_results: The original main query results
            full_path: The full path to the relation (e.g., "product__master")
            related_model: The related model class
            parent_pks: List of parent PKs (not used but kept for consistency)

        Returns:
            Dict mapping related object PK to its data dict
        """
        related_pk_name = related_model._meta.pk.name
        prefix = f"{full_path}__"

        result: dict[Any, dict] = {}

        for row in main_results:
            # Extract all fields with this path prefix
            related_dict: dict[str, Any] = {}
            for key, value in row.items():
                if key.startswith(prefix):
                    field_name = key[len(prefix) :]
                    # Skip further nested relations (contain another __)
                    if "__" not in field_name:
                        related_dict[field_name] = value

            # Get the related object's PK
            related_pk = related_dict.get(related_pk_name)
            if related_pk is not None and related_pk not in result:
                result[related_pk] = related_dict

        return result

    def _map_nested_fk_to_parents(
        self,
        main_results: list[dict],
        full_path: str,
        related_model: type[Model],
        parent_pks: list[Any],
        related_data: dict[Any, dict],
    ) -> dict[Any, dict | None]:
        """Map nested FK data back to parent PKs using main_results.

        When using select_related, we need to map parent objects to their
        related FK objects using the data in main_results.

        Args:
            main_results: The original main query results
            full_path: The full path to the relation (e.g., "product__master")
            related_model: The related model class
            parent_pks: List of parent PKs
            related_data: Dict of related PK -> data (may include nested relations)

        Returns:
            Dict mapping parent PK to related object data (or None)
        """
        related_pk_name = related_model._meta.pk.name
        pk_field_key = f"{full_path}__{related_pk_name}"

        # Build mapping from parent_pk to related_pk
        # We need to find the parent PK in main_results and map it to the related PK
        # The parent is one level up from full_path

        # Parse the path to find parent path
        path_parts = full_path.split("__")
        if len(path_parts) == 1:
            # Direct relation from main model - use main model's PK
            main_pk_name = self.model._meta.pk.name
            parent_to_related: dict[Any, Any] = {}
            for row in main_results:
                parent_pk = row.get(main_pk_name)
                related_pk = row.get(pk_field_key)
                if parent_pk is not None:
                    parent_to_related[parent_pk] = related_pk
        else:
            # Nested relation - parent is the previous level
            parent_path = "__".join(path_parts[:-1])
            parent_model = self._get_related_model_for_path(parent_path)
            if parent_model is None:
                return dict.fromkeys(parent_pks)
            parent_pk_name = parent_model._meta.pk.name
            parent_pk_field = f"{parent_path}__{parent_pk_name}"

            parent_to_related = {}
            for row in main_results:
                parent_pk = row.get(parent_pk_field)
                related_pk = row.get(pk_field_key)
                if parent_pk is not None:
                    parent_to_related[parent_pk] = related_pk

        # Map parent_pks to related data
        result: dict[Any, dict | None] = {}
        for parent_pk in parent_pks:
            related_pk = parent_to_related.get(parent_pk)
            if related_pk is not None and related_pk in related_data:
                result[parent_pk] = dict(related_data[related_pk])
            else:
                result[parent_pk] = None

        return result

    def _add_nested_relations(  # noqa: PLR0913
        self,
        model: type,
        data: dict[Any, dict],
        nested_relations: list[str],
        parent_pks: list[Any],
        main_results: list[dict] | None = None,
        parent_path: str = "",
    ) -> None:
        """Add nested relation data to already-fetched data.

        Args:
            model: The model class for the current level
            data: Dict of pk -> row data for items at this level
            nested_relations: List of nested relation paths to fetch
            parent_pks: List of primary keys for parent items
            main_results: Original main query results (for extracting select_related data)
            parent_path: Path prefix for tracking position in relation chain (e.g., "product__")
        """
        for nested_rel in nested_relations:
            parts = nested_rel.split("__", 1)
            rel_name = parts[0]
            further_nested = [parts[1]] if len(parts) > 1 else []

            try:
                field = model._meta.get_field(rel_name)  # type: ignore[union-attr]
            except FieldDoesNotExist:
                continue

            nested_data = self._fetch_nested_relation(
                model,
                field,
                rel_name,
                further_nested,
                parent_pks,
                main_results,
                parent_path,
            )

            for pk, row in data.items():
                row[rel_name] = nested_data.get(pk, [] if self._is_many_relation(field) else None)

    def _fetch_nested_relation(  # noqa: PLR0913
        self,
        parent_model: type,
        field: Any,
        relation_name: str,
        further_nested: list[str],
        parent_pks: list[Any],
        main_results: list[dict] | None = None,
        parent_path: str = "",
    ) -> dict[Any, list[dict] | dict | None]:
        """Fetch a nested relation for already-fetched data.

        Args:
            parent_model: The parent model class
            field: The field object for the relation
            relation_name: Name of the relation
            further_nested: List of further nested relations to fetch
            parent_pks: List of parent primary keys
            main_results: Original main query results (for extracting select_related data)
            parent_path: Path prefix for tracking position in relation chain
        """
        if isinstance(field, ManyToManyField):
            return self._fetch_nested_m2m(parent_model, field, further_nested, parent_pks, main_results, parent_path)  # type: ignore[return-value]
        if isinstance(field, ManyToOneRel):
            return self._fetch_nested_reverse_fk(
                parent_model,
                field,
                further_nested,
                parent_pks,
                main_results,
                parent_path,
            )  # type: ignore[return-value]
        if isinstance(field, ForeignKey):
            return self._fetch_nested_fk(parent_model, field, further_nested, parent_pks, main_results, parent_path)  # type: ignore[return-value]
        if isinstance(field, ManyToManyRel):
            return self._fetch_nested_reverse_m2m(
                parent_model,
                field,
                further_nested,
                parent_pks,
                main_results,
                parent_path,
            )  # type: ignore[return-value]
        if isinstance(field, GenericRelation):
            return self._fetch_nested_generic_relation(
                parent_model,
                field,
                further_nested,
                parent_pks,
                main_results,
                parent_path,
            )  # type: ignore[return-value]
        return {}

    def _fetch_nested_m2m(  # noqa: PLR0913
        self,
        parent_model: type,
        field: ManyToManyField,
        further_nested: list[str],
        parent_pks: list[Any],
        main_results: list[dict] | None = None,
        parent_path: str = "",
    ) -> dict[Any, list[dict]]:
        """Fetch nested ManyToMany data."""
        related_model = field.related_model
        through_model = field.remote_field.through

        source_col = field.m2m_column_name()
        target_col = field.m2m_reverse_name()

        through_qs = through_model.objects.filter(**{f"{source_col}__in": parent_pks})  # type: ignore[union-attr]
        through_data = list(through_qs.values(source_col, target_col))

        if not through_data:
            return {pk: [] for pk in parent_pks}

        related_pks = [t[target_col] for t in through_data]
        fetch_fields = [f.name for f in related_model._meta.concrete_fields]
        related_pk_name = related_model._meta.pk.name

        related_qs = related_model._default_manager.filter(pk__in=related_pks)
        related_data = {r[related_pk_name]: dict(r) for r in related_qs.values(*fetch_fields)}

        if further_nested:
            # Build full path for this relation
            full_path = f"{parent_path}{field.name}"
            new_parent_path = f"{full_path}__"
            self._add_nested_relations(
                related_model,
                related_data,
                further_nested,
                list(related_data.keys()),
                main_results,
                new_parent_path,
            )

        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for through_row in through_data:
            parent_pk = through_row[source_col]
            related_pk = through_row[target_col]
            if related_pk in related_data:
                result[parent_pk].append(dict(related_data[related_pk]))

        return result

    def _fetch_nested_reverse_fk(  # noqa: PLR0913
        self,
        parent_model: type,
        field: ManyToOneRel,
        further_nested: list[str],
        parent_pks: list[Any],
        main_results: list[dict] | None = None,
        parent_path: str = "",
    ) -> dict[Any, list[dict]]:
        """Fetch nested reverse ForeignKey data."""
        related_model = field.related_model
        fk_field_name = field.field.name

        fetch_fields = [f.name for f in related_model._meta.concrete_fields]
        related_qs = related_model._default_manager.filter(**{f"{fk_field_name}__in": parent_pks})
        related_data = list(related_qs.values(*fetch_fields))

        if further_nested and related_data:
            related_pk_name = related_model._meta.pk.name
            related_pks = [r[related_pk_name] for r in related_data]
            nested_dict = {r[related_pk_name]: dict(r) for r in related_data}
            # Build full path for this relation (use related_name from reverse FK)
            rel_name = field.get_accessor_name()
            full_path = f"{parent_path}{rel_name}"
            new_parent_path = f"{full_path}__"
            self._add_nested_relations(
                related_model,
                nested_dict,
                further_nested,
                related_pks,
                main_results,
                new_parent_path,
            )
            related_data = list(nested_dict.values())

        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row[fk_field_name]
            row_dict = dict(row)
            row_dict.pop(fk_field_name, None)
            result[parent_pk].append(row_dict)

        return result

    def _fetch_nested_fk(  # noqa: PLR0913
        self,
        parent_model: type,
        field: ForeignKey,
        further_nested: list[str],
        parent_pks: list[Any],
        main_results: list[dict] | None = None,
        parent_path: str = "",
    ) -> dict[Any, dict | None]:
        """Fetch nested ForeignKey data.

        Args:
            parent_model: The parent model class
            field: The ForeignKey field
            further_nested: List of further nested relations to fetch
            parent_pks: List of parent primary keys
            main_results: Original main query results (for extracting select_related data)
            parent_path: Path prefix for tracking position in relation chain (e.g., "product__")
        """
        related_model = field.related_model
        fk_attr = field.name
        related_pk_name = related_model._meta.pk.name

        # Build the full path for this relation (e.g., "product__master")
        full_path = f"{parent_path}{fk_attr}"

        # Check if this relation was already fetched via select_related
        select_related_fields = self._get_select_related_fields()
        if full_path in select_related_fields and main_results is not None:
            # Data already in main_results from JOIN - extract it
            related_data = self._extract_nested_fk_from_main_results(
                main_results,
                full_path,
                related_model,
                parent_pks,
            )
            fk_values = list(related_data.keys())
        else:
            # Not select_related - need to query
            fk_column = f"{fk_attr}_id"
            parent_qs = parent_model._default_manager.filter(pk__in=parent_pks)  # type: ignore[union-attr]
            pk_name = parent_model._meta.pk.name  # type: ignore[union-attr]
            fk_data = {r[pk_name]: r[fk_column] for r in parent_qs.values(pk_name, fk_column)}

            fk_values = list({v for v in fk_data.values() if v is not None})
            if not fk_values:
                return dict.fromkeys(parent_pks)

            fetch_fields = [f.name for f in related_model._meta.concrete_fields]

            related_qs = related_model._default_manager.filter(pk__in=fk_values)
            related_data = {r[related_pk_name]: dict(r) for r in related_qs.values(*fetch_fields)}

        if not related_data:
            return dict.fromkeys(parent_pks)

        if further_nested:
            # Pass main_results and updated path for further nesting
            new_parent_path = f"{full_path}__"
            self._add_nested_relations(
                related_model,
                related_data,
                further_nested,
                fk_values,
                main_results,
                new_parent_path,
            )

        # For select_related case, we need to map parent_pk -> related data differently
        if full_path in select_related_fields and main_results is not None:
            # Map parent_pk to related object using main_results
            return self._map_nested_fk_to_parents(
                main_results,
                full_path,
                related_model,
                parent_pks,
                related_data,
            )

        # For non-select_related case, use fk_data mapping
        result: dict[Any, dict | None] = {}
        for parent_pk in parent_pks:
            fk_value = fk_data.get(parent_pk)
            result[parent_pk] = dict(related_data[fk_value]) if fk_value in related_data else None

        return result

    def _fetch_nested_reverse_m2m(  # noqa: PLR0913
        self,
        parent_model: type,
        field: ManyToManyRel,
        further_nested: list[str],
        parent_pks: list[Any],
        main_results: list[dict] | None = None,
        parent_path: str = "",
    ) -> dict[Any, list[dict]]:
        """Fetch nested reverse ManyToMany data."""
        related_model = field.related_model
        through_model = field.through
        m2m_field = field.field

        source_col = m2m_field.m2m_reverse_name()
        target_col = m2m_field.m2m_column_name()

        through_qs = through_model.objects.filter(**{f"{source_col}__in": parent_pks})  # type: ignore[union-attr]
        through_data = list(through_qs.values(source_col, target_col))

        if not through_data:
            return {pk: [] for pk in parent_pks}

        related_pks = [t[target_col] for t in through_data]
        fetch_fields = [f.name for f in related_model._meta.concrete_fields]
        related_pk_name = related_model._meta.pk.name

        related_qs = related_model._default_manager.filter(pk__in=related_pks)
        related_data = {r[related_pk_name]: dict(r) for r in related_qs.values(*fetch_fields)}

        if further_nested:
            # Build full path for this relation
            rel_name = field.get_accessor_name()
            full_path = f"{parent_path}{rel_name}"
            new_parent_path = f"{full_path}__"
            self._add_nested_relations(
                related_model,
                related_data,
                further_nested,
                list(related_data.keys()),
                main_results,
                new_parent_path,
            )

        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for through_row in through_data:
            parent_pk = through_row[source_col]
            related_pk = through_row[target_col]
            if related_pk in related_data:
                result[parent_pk].append(dict(related_data[related_pk]))

        return result

    def _fetch_nested_generic_relation(  # noqa: PLR0913
        self,
        parent_model: type,
        field: GenericRelation,
        further_nested: list[str],
        parent_pks: list[Any],
        main_results: list[dict] | None = None,
        parent_path: str = "",
    ) -> dict[Any, list[dict]]:
        """Fetch nested GenericRelation data."""
        related_model = field.related_model
        ct_field_name = field.content_type_field_name
        obj_id_field_name = field.object_id_field_name

        # Get the ContentType for the parent model
        parent_ct = ContentType.objects.get_for_model(cast("type[Model]", parent_model))

        fetch_fields = [f.name for f in related_model._meta.concrete_fields]
        related_pk_name = related_model._meta.pk.name

        # Ensure object_id is in fetch_fields for grouping
        if obj_id_field_name not in fetch_fields:
            fetch_fields = [*fetch_fields, obj_id_field_name]

        related_qs = related_model._default_manager.filter(
            **{ct_field_name: parent_ct, f"{obj_id_field_name}__in": parent_pks},
        )
        related_data = list(related_qs.values(*fetch_fields))

        if further_nested and related_data:
            related_pks = [r[related_pk_name] for r in related_data]
            nested_dict = {r[related_pk_name]: dict(r) for r in related_data}
            # Build full path for this relation
            full_path = f"{parent_path}{field.name}"
            new_parent_path = f"{full_path}__"
            self._add_nested_relations(
                related_model,
                nested_dict,
                further_nested,
                related_pks,
                main_results,
                new_parent_path,
            )
            related_data = list(nested_dict.values())

        # Group by parent PK (object_id)
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row[obj_id_field_name]
            row_dict = dict(row)
            # Remove the object_id field from output
            row_dict.pop(obj_id_field_name, None)
            # Also remove content_type_id if present
            row_dict.pop(f"{ct_field_name}_id", None)
            result[parent_pk].append(row_dict)

        return result

    def _is_many_relation(self, field: Any) -> bool:
        """Check if a field represents a many-relation."""
        return isinstance(field, ManyToManyField | ManyToManyRel | ManyToOneRel | GenericRelation)


class NestedValuesQuerySet(NestedValuesQuerySetMixin[_ModelT_co], QuerySet[_ModelT_co, _ModelT_co]):
    """QuerySet that adds .values_nested() for nested dictionaries.

    This is a ready-to-use QuerySet combining NestedValuesQuerySetMixin with Django's QuerySet.

    Usage:
        class Book(models.Model):
            objects = NestedValuesQuerySet.as_manager()

        # Use standard Django patterns:
        Book.objects.only("title").select_related("publisher").prefetch_related("authors").values_nested()
        # Returns: [{'id': 1, 'title': '...', 'publisher': {...}, 'authors': [...]}, ...]

    If you have a custom QuerySet, use NestedValuesQuerySetMixin instead:

        class MyQuerySet(NestedValuesQuerySetMixin, QuerySet):
            def my_custom_method(self):
                ...

    Type hints: The queryset is generic over the model type. After calling values_nested(),
    it yields dict[str, Any] when iterated, similar to Django's values() method.
    """
