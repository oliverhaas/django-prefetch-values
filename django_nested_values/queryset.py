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


def _build_dict_from_klass_info(
    row: tuple[Any, ...],
    klass_info: dict[str, Any],
    select: list[tuple[Any, ...]],
) -> dict[str, Any]:
    """Build a dict for a model from a row using klass_info metadata.

    This uses Django's internal compiler metadata to know exactly which
    columns belong to which model, avoiding manual field path parsing.

    Args:
        row: A tuple of values from the database row
        klass_info: The klass_info dict from compiler, containing:
            - 'model': the model class
            - 'select_fields': list of column indices for this model
            - 'related_klass_infos': list of klass_info for select_related models
        select: The compiler.select list, mapping indices to column expressions

    Returns:
        A dict with field names as keys and values from the row
    """
    result: dict[str, Any] = {}

    # Extract fields for this model using select_fields indices
    for idx in klass_info["select_fields"]:
        col_expr = select[idx][0]
        # Get the field name (attname gives us 'publisher_id' for FKs)
        field_name = col_expr.target.attname
        result[field_name] = row[idx]

    # Process related models (from select_related)
    for related_ki in klass_info.get("related_klass_infos", []):
        # Get the relation name from the field
        relation_name = related_ki["field"].name

        # Check if the related object is NULL by checking if PK is None
        # The first select_field is typically the PK
        pk_idx = related_ki["select_fields"][0]
        if row[pk_idx] is None:
            # Related object doesn't exist (NULL FK)
            continue

        # Recursively build the nested dict
        nested_dict = _build_dict_from_klass_info(row, related_ki, select)
        result[relation_name] = nested_dict

    return result


def _execute_queryset_as_dicts(
    queryset: QuerySet,
    db: str,
) -> list[dict[str, Any]]:
    """Execute a queryset using the compiler and return results as nested dicts.

    This uses Django's internal compiler to execute the query and builds
    nested dictionaries directly from the raw row tuples - NO model instantiation.

    The pipeline is: raw SQL rows → dicts (via klass_info metadata)

    This handles:
    - select_related: Automatically builds nested dicts for related objects
    - only()/defer(): Compiler respects deferred field loading
    - Custom querysets: Works with any queryset (base, filtered, etc.)

    Args:
        queryset: The queryset to execute (may have select_related, only, etc.)
        db: Database alias to use

    Returns:
        List of nested dictionaries, one per row
    """
    compiler = queryset.query.get_compiler(using=db)
    results = compiler.execute_sql()

    if results is None:
        return []

    select = compiler.select
    klass_info = compiler.klass_info

    if klass_info is None:
        return []

    return [_build_dict_from_klass_info(row, klass_info, select) for row in compiler.results_iter(results)]


def _execute_prefetch_as_dicts(
    queryset: QuerySet,
    db: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute a prefetch queryset and return (dicts, extra_values).

    This is similar to _execute_queryset_as_dicts but also extracts the extra
    columns that Django adds for prefetch grouping (like _prefetch_related_val_*).

    Returns:
        Tuple of (list of model dicts, list of extra column dicts for grouping)
    """
    compiler = queryset.query.get_compiler(using=db)
    results = compiler.execute_sql()

    if results is None:
        return [], []

    select = compiler.select
    klass_info = compiler.klass_info

    if klass_info is None:
        return [], []

    # Find extra column indices (columns with _prefetch_related_val_* alias)
    # Extra columns from .extra() have format: (RawSQL, (sql, params), alias)
    extra_indices = [
        (i, s[2])
        for i, s in enumerate(select)
        if len(s) >= 3 and s[2] and s[2].startswith("_prefetch_related_val_")  # noqa: PLR2004
    ]

    dicts = []
    extra_values = []

    for row in compiler.results_iter(results):
        # Build the model dict from klass_info
        row_dict = _build_dict_from_klass_info(row, klass_info, select)

        # Extract extra column values for grouping
        extra_vals = {alias: row[idx] for idx, alias in extra_indices}

        dicts.append(row_dict)
        extra_values.append(extra_vals)

    return dicts, extra_values


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
        db = queryset.db
        prefetch_lookups = queryset._prefetch_related_lookups  # type: ignore[attr-defined]

        # Build the queryset (applies select_related, only/defer, etc.)
        main_qs = queryset._build_main_queryset()

        # When using select_related with deferred loading, we need to ensure the FK
        # fields for relations are NOT deferred. Otherwise Django raises:
        # "Field X.y cannot be both deferred and traversed using select_related"
        self._ensure_fk_fields_not_deferred(main_qs)

        # Get the compiler - it knows the complete structure including select_related
        compiler = main_qs.query.get_compiler(using=db)

        # Execute the query
        results = compiler.execute_sql(
            chunked_fetch=self.chunked_fetch,
            chunk_size=self.chunk_size,
        )
        if results is None:
            return

        # Get metadata for building nested dicts
        select = compiler.select
        klass_info = compiler.klass_info

        if klass_info is None:
            return

        # Build nested dicts directly from rows using klass_info
        main_results = [_build_dict_from_klass_info(row, klass_info, select) for row in compiler.results_iter(results)]

        if not main_results:
            return

        # If no prefetch, just yield the results
        if not prefetch_lookups:
            yield from main_results
            return

        # Get PKs for prefetch queries
        pk_name = queryset.model._meta.pk.name
        pk_values = [r[pk_name] for r in main_results]

        # Fetch prefetched relations
        prefetched_data = queryset._fetch_all_prefetched(prefetch_lookups, pk_values, main_results)

        # Merge prefetched data into results and yield
        for row in main_results:
            pk = row[pk_name]
            for attr_name, data_by_pk in prefetched_data.items():
                prefetch_value = data_by_pk.get(pk, [])
                self._set_nested_value(row, attr_name, prefetch_value)
            yield row

    def _set_nested_value(self, row: dict, attr_path: str, value: Any) -> None:
        """Set a value in a nested dict using a path like 'publisher__books'.

        Navigates to the correct nested location and sets or merges the value.
        """
        parts = attr_path.split("__")
        target = row

        # Navigate to the parent dict
        for part in parts[:-1]:
            if part in target and isinstance(target[part], dict):
                target = target[part]
            else:
                # Path doesn't exist or isn't a dict, set at top level
                row[attr_path] = value
                return

        # Set the final value
        final_key = parts[-1]
        if final_key in target:
            existing = target[final_key]
            if isinstance(existing, dict) and isinstance(value, dict):
                # Recursively merge dicts
                self._merge_dicts(existing, value)
            # Otherwise keep existing (select_related takes precedence)
        else:
            target[final_key] = value

    def _merge_dicts(self, target: dict, source: dict) -> None:
        """Recursively merge source dict into target dict.

        New keys from source are added. For existing keys where both values
        are dicts, recursively merge. Otherwise, target value takes precedence
        (select_related data wins over prefetch data for same field).
        """
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dicts(target[key], value)

    def _ensure_fk_fields_not_deferred(self, qs: QuerySet) -> None:
        """Ensure FK fields for select_related are not deferred.

        When using .only() without the FK field and then select_related(),
        Django would raise an error. This method clears deferral for FK fields
        that are traversed via select_related.
        """
        select_related = qs.query.select_related
        if not select_related:
            return

        deferred_fields, is_defer = qs.query.deferred_loading
        if not deferred_fields:
            return

        # Get FK field names for select_related relations
        if select_related is True:
            # select_related() with no args - need all FK fields
            fk_fields = {f.attname for f in qs.model._meta.concrete_fields if isinstance(f, ForeignKey)}
        else:
            # select_related is a dict like {'publisher': {}, 'author': {'publisher': {}}}
            # We need the FK attnames for top-level relations
            fk_fields = set()
            for relation_name in select_related:
                try:
                    field = qs.model._meta.get_field(relation_name)
                    if isinstance(field, ForeignKey):
                        fk_fields.add(field.attname)
                except FieldDoesNotExist:
                    pass

        if not fk_fields:
            return

        # Modify deferred loading to include FK fields
        if is_defer:
            # .defer() was used - remove FK fields from deferred set
            new_deferred = deferred_fields - fk_fields
            qs.query.deferred_loading = (new_deferred, True)
        else:
            # .only() was used - add FK fields to included set
            new_only = deferred_fields | fk_fields
            qs.query.deferred_loading = (new_only, False)


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
        relation_name = (
            lookup.prefetch_through.split("__")[0] if isinstance(lookup, Prefetch) else lookup.split("__")[0]
        )

        try:
            field = self.model._meta.get_field(relation_name)
        except FieldDoesNotExist:
            return {}

        custom_qs = lookup.queryset if isinstance(lookup, Prefetch) and lookup.queryset is not None else None

        # Dispatch to internal methods directly
        if isinstance(field, ManyToManyField):
            return self._fetch_m2m_internal(
                cast("type[Model]", field.related_model),
                field.related_query_name(),
                field.name,
                nested_relations,
                parent_pks,
                custom_qs,
                main_results,
                "",
                field,
            )  # type: ignore[return-value]
        if isinstance(field, ManyToOneRel):
            return self._fetch_reverse_fk_internal(
                cast("type[Model]", field.related_model),
                field.field.name,
                field.get_accessor_name() or field.name,
                nested_relations,
                parent_pks,
                custom_qs,
                main_results,
                "",
            )  # type: ignore[return-value]
        if isinstance(field, ForeignKey):
            pk_name = self.model._meta.pk.name
            return self._fetch_fk_internal(
                field,
                nested_relations,
                parent_pks,
                {r[pk_name]: r for r in main_results},
                custom_qs,
                main_results,
                "",
            )  # type: ignore[return-value]
        if isinstance(field, ManyToManyRel):
            return self._fetch_m2m_internal(
                cast("type[Model]", field.related_model),
                field.field.name,
                field.get_accessor_name() or field.name,
                nested_relations,
                parent_pks,
                custom_qs,
                main_results,
                "",
                field,
            )  # type: ignore[return-value]
        if isinstance(field, GenericRelation):
            return self._fetch_generic_relation_internal(
                field,
                nested_relations,
                parent_pks,
                self.model,
                custom_qs,
                main_results,
                "",
            )  # type: ignore[return-value]

        return {}

    def _get_select_related_from_queryset(self, qs: QuerySet | None) -> dict[str, Any]:
        """Get select_related structure from a queryset.

        Returns a dict like {'publisher': {}, 'publisher__country': {}} for
        select_related('publisher', 'publisher__country').
        """
        if qs is None:
            return {}

        select_related = qs.query.select_related
        if not select_related:
            return {}

        if select_related is True:
            # select_related() with no args - get all FK fields
            result = {}
            for field in qs.model._meta.concrete_fields:
                if isinstance(field, ForeignKey):
                    result[field.name] = {}
            return result

        # select_related is a nested dict like {'publisher': {'country': {}}}
        # Flatten it to paths
        result = {}
        self._flatten_select_related_to_paths(select_related, "", result)
        return result

    def _flatten_select_related_to_paths(
        self,
        select_related: dict,
        prefix: str,
        result: dict[str, Any],
    ) -> None:
        """Flatten nested select_related dict to path-based dict."""
        for relation_name, nested in select_related.items():
            full_path = f"{prefix}{relation_name}" if prefix else relation_name
            result[full_path] = {}
            if nested:
                self._flatten_select_related_to_paths(nested, f"{full_path}__", result)

    def _fetch_m2m_internal(  # noqa: PLR0913
        self,
        related_model: type[Model],
        accessor: str,
        relation_name: str,
        nested_relations: list[str],
        parent_pks: list[Any],
        custom_qs: QuerySet | None,
        main_results: list[dict] | None,
        parent_path: str,
        m2m_field: ManyToManyField | ManyToManyRel,
    ) -> dict[Any, list[dict]]:
        """Internal helper to fetch M2M data - used by forward/reverse M2M, top-level and nested.

        Uses Django's compiler with .extra() annotation to match Django's native
        prefetch_related behavior. Raw rows → dicts, no model instantiation.
        """
        from django.db import connections

        related_pk_name = related_model._meta.pk.name

        # Build queryset - filter related model where accessor contains parent PKs
        if custom_qs is not None:
            related_qs = custom_qs.filter(**{f"{accessor}__in": parent_pks})
        else:
            related_qs = related_model._default_manager.filter(**{f"{accessor}__in": parent_pks})

        # Get the actual ManyToManyField (for ManyToManyRel, it's in .field)
        actual_field = m2m_field.field if isinstance(m2m_field, ManyToManyRel) else m2m_field

        # Get through table info and add extra select for parent FK (matching Django's approach)
        through_model = actual_field.remote_field.through
        assert through_model is not None  # noqa: S101
        # Determine which FK to use: Forward M2M uses m2m_field_name(), Reverse M2M uses m2m_reverse_name()
        source_field_name = (
            actual_field.m2m_reverse_name() if isinstance(m2m_field, ManyToManyRel) else actual_field.m2m_field_name()
        )
        fk = cast("ForeignKey[Any, Any]", through_model._meta.get_field(source_field_name))
        join_table = through_model._meta.db_table
        qn = connections[self.db].ops.quote_name

        extra_select = {
            f"_prefetch_related_val_{f.attname}": f"{qn(join_table)}.{qn(f.column)}" for f in fk.local_related_fields
        }
        related_qs = related_qs.extra(select=extra_select)  # noqa: S610  # safe: identifiers quoted via qn()

        # Execute with compiler and extract model dicts + extra columns
        dicts, extra_values = _execute_prefetch_as_dicts(related_qs, self.db)

        if not dicts:
            return {pk: [] for pk in parent_pks}

        # Group by parent PK (from extra column)
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        related_data: dict[Any, dict] = {}

        for row_dict, extra_vals in zip(dicts, extra_values, strict=True):
            # Get parent PK from first extra column
            source_pk = next(iter(extra_vals.values())) if extra_vals else None
            if source_pk in result:
                result[source_pk].append(row_dict)

            # Track for nested relations
            related_pk = row_dict.get(related_pk_name)
            if nested_relations and related_pk not in related_data:
                related_data[related_pk] = row_dict

        # Handle nested relations
        if nested_relations and related_data:
            select_related_paths = self._get_select_related_from_queryset(custom_qs)
            remaining_nested = [rel for rel in nested_relations if rel.split("__")[0] not in select_related_paths]
            if remaining_nested:
                full_path = f"{parent_path}{relation_name}" if parent_path else relation_name
                self._add_nested_relations(
                    related_model,
                    related_data,
                    remaining_nested,
                    list(related_data.keys()),
                    main_results,
                    f"{full_path}__",
                )
                # Update results with nested data
                for items in result.values():
                    for item in items:
                        pk_val = item.get(related_pk_name)
                        if pk_val and pk_val in related_data:
                            for key, val in related_data[pk_val].items():
                                if key not in item or not isinstance(item[key], dict):
                                    item[key] = val

        return result

    def _fetch_reverse_fk_internal(  # noqa: PLR0913
        self,
        related_model: type[Model],
        fk_field_name: str,
        relation_name: str,
        nested_relations: list[str],
        parent_pks: list[Any],
        custom_qs: QuerySet | None,
        main_results: list[dict] | None,
        parent_path: str,
    ) -> dict[Any, list[dict]]:
        """Fetch reverse FK data using compiler. Raw rows → dicts, no model instantiation."""
        related_pk_name = related_model._meta.pk.name
        # Get the FK field's attname (e.g., 'book_id' for FK field 'book')
        fk_field = related_model._meta.get_field(fk_field_name)
        fk_attname = fk_field.attname  # type: ignore[union-attr]

        # Build queryset - custom_qs already has select_related if specified
        if custom_qs is not None:
            related_qs = custom_qs.filter(**{f"{fk_field_name}__in": parent_pks})
        else:
            related_qs = related_model._default_manager.filter(**{f"{fk_field_name}__in": parent_pks})

        # Execute with compiler - handles select_related automatically!
        # Raw rows → dicts, no model instantiation
        related_data = _execute_queryset_as_dicts(related_qs, self.db)

        if not related_data:
            return {pk: [] for pk in parent_pks}

        # Handle nested relations (prefetch within prefetch)
        if nested_relations:
            # Get select_related paths to filter out already-handled relations
            select_related_paths = self._get_select_related_from_queryset(custom_qs)
            remaining_nested = [rel for rel in nested_relations if rel.split("__")[0] not in select_related_paths]
            if remaining_nested:
                related_pks = [r[related_pk_name] for r in related_data]
                nested_dict = {r[related_pk_name]: r for r in related_data}
                full_path = f"{parent_path}{relation_name}" if parent_path else relation_name
                new_parent_path = f"{full_path}__"
                self._add_nested_relations(
                    related_model,
                    nested_dict,
                    remaining_nested,
                    related_pks,
                    main_results,
                    new_parent_path,
                )
                related_data = list(nested_dict.values())

        # Group by parent PK (FK value is in result as attname, e.g., 'book_id')
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row.get(fk_attname)
            if parent_pk in result:
                row_dict = dict(row)
                # Remove the FK attname from output (e.g., remove 'book_id')
                row_dict.pop(fk_attname, None)
                result[parent_pk].append(row_dict)

        return result

    def _fetch_fk_internal(  # noqa: PLR0913
        self,
        field: ForeignKey,
        nested_relations: list[str],
        parent_pks: list[Any],
        parent_data: dict[Any, dict] | None,
        custom_qs: QuerySet | None,
        main_results: list[dict] | None,
        parent_path: str,
        parent_model: type[Model] | None = None,
    ) -> dict[Any, dict | None]:
        """Fetch FK data using compiler. Raw rows → dicts, no model instantiation."""
        related_model = field.related_model
        fk_attname = field.attname  # e.g., publisher_id
        relation_name = field.name  # e.g., publisher
        related_pk_name = related_model._meta.pk.name

        # Extract FK values from parent_data or query if not available
        fk_data: dict[Any, Any] = {}
        if parent_data is not None:
            for pk, row in parent_data.items():
                fk_value = row.get(fk_attname)
                if fk_value is None:
                    fk_value = row.get(relation_name)
                    if isinstance(fk_value, dict):
                        fk_value = fk_value.get(related_pk_name)
                fk_data[pk] = fk_value
        elif parent_model is not None:
            # Fallback: query the database for FK values
            parent_qs = parent_model._default_manager.filter(pk__in=parent_pks)
            pk_name = parent_model._meta.pk.name
            fk_data = {r[pk_name]: r[fk_attname] for r in parent_qs.values(pk_name, fk_attname)}
        else:
            return dict.fromkeys(parent_pks)

        fk_values = list({v for v in fk_data.values() if v is not None})
        if not fk_values:
            return dict.fromkeys(parent_pks)

        # Check if data already exists via select_related (nested dict in parent_data)
        has_select_related = parent_data is not None and any(
            relation_name in row and isinstance(row.get(relation_name), dict) for row in parent_data.values()
        )

        if has_select_related and custom_qs is None:
            # Extract data from nested dicts - no query needed
            related_data: dict[Any, dict] = {}
            for row in parent_data.values():  # type: ignore[union-attr]
                nested = row.get(relation_name)
                if isinstance(nested, dict) and nested.get(related_pk_name) is not None:
                    related_data[nested[related_pk_name]] = dict(nested)
        else:
            # Query the related model using compiler - handles select_related automatically!
            if custom_qs is not None:
                related_qs = custom_qs.filter(pk__in=fk_values)
            else:
                related_qs = related_model._default_manager.filter(pk__in=fk_values)

            # Execute with compiler - raw rows → dicts, no model instantiation
            results = _execute_queryset_as_dicts(related_qs, self.db)
            related_data = {r[related_pk_name]: r for r in results}

        if not related_data:
            return dict.fromkeys(parent_pks)

        # Handle nested relations
        if nested_relations:
            full_path = f"{parent_path}{relation_name}__" if parent_path else f"{relation_name}__"
            self._add_nested_relations(
                related_model,
                related_data,
                nested_relations,
                fk_values,
                main_results,
                full_path,
            )

        # Map parent_pk to related data
        result: dict[Any, dict | None] = {}
        for parent_pk in parent_pks:
            fk_value = fk_data.get(parent_pk)
            result[parent_pk] = dict(related_data[fk_value]) if fk_value in related_data else None

        return result

    def _fetch_generic_relation_internal(  # noqa: PLR0913
        self,
        field: GenericRelation,
        nested_relations: list[str],
        parent_pks: list[Any],
        parent_model: type[Model],
        custom_qs: QuerySet | None,
        main_results: list[dict] | None,
        parent_path: str,
    ) -> dict[Any, list[dict]]:
        """Internal helper to fetch GenericRelation data - used by top-level and nested."""
        related_model = field.related_model
        ct_field_name = field.content_type_field_name
        obj_id_field_name = field.object_id_field_name
        related_pk_name = related_model._meta.pk.name

        parent_ct = ContentType.objects.get_for_model(parent_model)
        filter_kwargs = {ct_field_name: parent_ct, f"{obj_id_field_name}__in": parent_pks}

        if custom_qs is not None:
            related_qs = custom_qs.filter(**filter_kwargs)
        else:
            related_qs = related_model._default_manager.filter(**filter_kwargs)

        related_data = _execute_queryset_as_dicts(related_qs, self.db)

        if nested_relations and related_data:
            related_pks = [r[related_pk_name] for r in related_data]
            nested_dict = {r[related_pk_name]: r for r in related_data}
            full_path = f"{parent_path}{field.name}" if parent_path else field.name
            self._add_nested_relations(
                related_model,
                nested_dict,
                nested_relations,
                related_pks,
                main_results,
                f"{full_path}__",
            )
            related_data = list(nested_dict.values())

        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row[obj_id_field_name]
            row_dict = dict(row)
            row_dict.pop(obj_id_field_name, None)
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

            # Fetch the related objects using compiler - no model instantiation
            related_qs = qs.filter(pk__in=object_ids)
            results = _execute_queryset_as_dicts(related_qs, self.db)
            related_data = {r[related_pk_name]: r for r in results}

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

    def _add_nested_relations(  # noqa: PLR0913
        self,
        model: type[Model],
        data: dict[Any, dict],
        nested_relations: list[str],
        parent_pks: list[Any],
        main_results: list[dict] | None = None,
        parent_path: str = "",
    ) -> None:
        """Fetch and add nested relation data to already-fetched parent data."""
        for nested_rel in nested_relations:
            parts = nested_rel.split("__", 1)
            rel_name = parts[0]
            further_nested = [parts[1]] if len(parts) > 1 else []

            try:
                field = model._meta.get_field(rel_name)
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
                data,  # Pass already-fetched parent data
            )

            for pk, row in data.items():
                row[rel_name] = nested_data.get(pk, [] if self._is_many_relation(field) else None)

    def _fetch_nested_relation(  # noqa: PLR0913
        self,
        parent_model: type[Model],
        field: Any,
        relation_name: str,
        further_nested: list[str],
        parent_pks: list[Any],
        main_results: list[dict] | None = None,
        parent_path: str = "",
        parent_data: dict[Any, dict] | None = None,
    ) -> dict[Any, list[dict] | dict | None]:
        """Fetch a nested relation for already-fetched data."""
        if isinstance(field, ManyToManyField):
            return self._fetch_m2m_internal(
                related_model=field.related_model,
                accessor=field.related_query_name(),
                relation_name=field.name,
                nested_relations=further_nested,
                parent_pks=parent_pks,
                custom_qs=None,
                main_results=main_results,
                parent_path=parent_path,
                m2m_field=field,
            )  # type: ignore[return-value]
        if isinstance(field, ManyToOneRel):
            return self._fetch_reverse_fk_internal(
                related_model=field.related_model,
                fk_field_name=field.field.name,
                relation_name=field.get_accessor_name() or field.name,
                nested_relations=further_nested,
                parent_pks=parent_pks,
                custom_qs=None,
                main_results=main_results,
                parent_path=parent_path,
            )  # type: ignore[return-value]
        if isinstance(field, ForeignKey):
            return self._fetch_fk_internal(
                field=field,
                nested_relations=further_nested,
                parent_pks=parent_pks,
                parent_data=parent_data,
                custom_qs=None,
                main_results=main_results,
                parent_path=parent_path,
                parent_model=parent_model,
            )  # type: ignore[return-value]
        if isinstance(field, ManyToManyRel):
            return self._fetch_m2m_internal(
                related_model=field.related_model,
                accessor=field.field.name,
                relation_name=field.get_accessor_name() or field.name,
                nested_relations=further_nested,
                parent_pks=parent_pks,
                custom_qs=None,
                main_results=main_results,
                parent_path=parent_path,
                m2m_field=field,
            )  # type: ignore[return-value]
        if isinstance(field, GenericRelation):
            return self._fetch_generic_relation_internal(
                field,
                further_nested,
                parent_pks,
                parent_model,
                None,
                main_results,
                parent_path,
            )  # type: ignore[return-value]
        return {}

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
