"""Custom QuerySet that adds .values_nested() for nested dictionaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, Self, TypeVar, cast, overload

from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.prefetch import GenericPrefetch
from django.core.exceptions import FieldDoesNotExist
from django.db import connections
from django.db.models import ForeignKey, ManyToManyField, ManyToManyRel, ManyToOneRel, Model, Prefetch, QuerySet
from django.db.models.query import BaseIterable

# TypeVar for the model type, used for generic typing with django-stubs
_ModelT_co = TypeVar("_ModelT_co", bound=Model, covariant=True)

# Type alias for dict-like container classes
_ContainerType = type[dict[str, Any]]


class AttrDict(dict[str, Any]):
    """Dict subclass with attribute access - minimal overhead.

    Inherits from dict so isinstance(x, dict) is True and all dict
    operations work unchanged. Just adds __getattr__ for dot access.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        """Get item as attribute."""
        try:
            return self[name]
        except KeyError:
            msg = f"'{type(self).__name__}' object has no attribute '{name}'"
            raise AttributeError(msg) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """Set item as attribute."""
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete item as attribute."""
        try:
            del self[name]
        except KeyError:
            msg = f"'{type(self).__name__}' object has no attribute '{name}'"
            raise AttributeError(msg) from None


if TYPE_CHECKING:
    from collections.abc import Iterator

    # For type checking, pretend the mixin inherits from QuerySet
    # This allows type checkers to see QuerySet methods on the mixin
    class _MixinBase(QuerySet[_ModelT_co, _ModelT_co]):
        pass
else:
    # At runtime, use Generic to allow subscripting like NestedValuesQuerySetMixin[Book]
    _MixinBase = Generic


def _build_from_klass_info(
    row: tuple[Any, ...],
    klass_info: dict[str, Any],
    select: list[tuple[Any, ...]],
    container: _ContainerType = dict,
) -> dict[str, Any]:
    """Build a dict (or dict subclass like AttrDict) directly from a row.

    This uses Django's internal compiler metadata to know exactly which
    columns belong to which model, avoiding manual field path parsing.

    Args:
        row: A tuple of values from the database row
        klass_info: The klass_info dict from compiler
        select: The compiler.select list
        container: The dict-like class to use (dict or AttrDict)

    Returns:
        A container instance with field names as keys

    """
    # Build directly into the container - no intermediate dict
    result = container()

    for idx in klass_info["select_fields"]:
        col_expr = select[idx][0]
        result[col_expr.target.attname] = row[idx]

    for related_ki in klass_info.get("related_klass_infos", []):
        pk_idx = related_ki["select_fields"][0]
        if row[pk_idx] is None:
            continue
        result[related_ki["field"].name] = _build_from_klass_info(row, related_ki, select, container)

    return result


def _execute_queryset(
    queryset: QuerySet[Any, Any],
    db: str,
    container: _ContainerType = dict,
) -> list[dict[str, Any]]:
    """Execute a queryset and return results as nested dicts (or AttrDict).

    Args:
        queryset: The queryset to execute
        db: Database alias to use
        container: The dict-like class to use (dict or AttrDict)

    Returns:
        List of nested containers, one per row

    """
    compiler = queryset.query.get_compiler(using=db)
    results = compiler.execute_sql()

    if results is None:
        return []

    select = compiler.select
    klass_info = compiler.klass_info

    if klass_info is None:
        return []

    return [_build_from_klass_info(row, klass_info, select, container) for row in compiler.results_iter(results)]


def _execute_prefetch(
    queryset: QuerySet[Any, Any],
    db: str,
    container: _ContainerType = dict,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute a prefetch queryset and return (containers, extra_values).

    This extracts the extra columns Django adds for prefetch grouping.

    Args:
        queryset: The queryset to execute
        db: Database alias to use
        container: The dict-like class to use (dict or AttrDict)

    Returns:
        Tuple of (list of model containers, list of extra column dicts for grouping)

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
    extra_indices = [
        (i, s[2])
        for i, s in enumerate(select)
        if len(s) >= 3 and s[2] and s[2].startswith("_prefetch_related_val_")  # noqa: PLR2004
    ]

    containers = []
    extra_values = []

    for row in compiler.results_iter(results):
        row_container = _build_from_klass_info(row, klass_info, select, container)
        extra_vals = {alias: row[idx] for idx, alias in extra_indices}
        containers.append(row_container)
        extra_values.append(extra_vals)

    return containers, extra_values


class NestedValuesIterable(BaseIterable):  # type: ignore[type-arg]
    """Iterable that yields nested dictionaries for QuerySet.values_nested().

    This follows Django's pattern of using iterable classes (like ValuesIterable)
    to control how queryset iteration yields results.
    """

    if TYPE_CHECKING:
        # The queryset is expected to be a NestedValuesQuerySetMixin
        queryset: NestedValuesQuerySetMixin[Any]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the queryset, yielding nested dictionaries (or AttrDict)."""
        queryset = self.queryset
        db = queryset.db
        prefetch_lookups = getattr(queryset, "_nested_prefetch_lookups", ())

        main_qs = queryset._build_main_queryset()
        self._ensure_fk_fields_not_deferred(main_qs)

        compiler = main_qs.query.get_compiler(using=db)
        results = compiler.execute_sql(
            chunked_fetch=self.chunked_fetch,
            chunk_size=self.chunk_size,
        )
        if results is None:
            return

        select = compiler.select
        klass_info = compiler.klass_info
        if klass_info is None:
            return

        # Choose container based on as_attr_dicts flag
        as_attr_dicts = getattr(queryset, "_as_attr_dicts", False)
        container: _ContainerType = AttrDict if as_attr_dicts else dict  # type: ignore[assignment]
        pk_name = queryset.model._meta.pk.name

        # Build main results - unified path for both dict and AttrDict
        main_results = [
            _build_from_klass_info(row, klass_info, select, container) for row in compiler.results_iter(results)
        ]
        if not main_results:
            return

        if not prefetch_lookups:
            yield from main_results
            return

        # Fetch prefetched data and attach to main results
        pk_values = [r[pk_name] for r in main_results]
        prefetched_data = queryset._fetch_all_prefetched(
            prefetch_lookups,
            pk_values,
            main_results,
            container=container,
        )

        for row in main_results:
            pk = row[pk_name]
            for attr_name, data_by_pk in prefetched_data.items():
                prefetch_value = data_by_pk.get(pk, [])
                self._set_nested_value(row, attr_name, prefetch_value)
            yield row

    def _set_nested_value(self, row: dict[str, Any], attr_path: str, value: Any) -> None:
        """Set a value in a nested dict using a path like 'publisher__books'."""
        parts = attr_path.split("__")
        target = row

        for part in parts[:-1]:
            if part in target and isinstance(target[part], dict):
                target = target[part]
            else:
                row[attr_path] = value
                return

        final_key = parts[-1]
        if final_key in target:
            existing = target[final_key]
            if isinstance(existing, dict) and isinstance(value, dict):
                self._merge_dicts(existing, value)
        else:
            target[final_key] = value

    def _merge_dicts(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        """Recursively merge source dict into target dict."""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dicts(target[key], value)

    def _ensure_fk_fields_not_deferred(self, qs: QuerySet[Any, Any]) -> None:
        """Ensure FK fields for select_related are not deferred."""
        select_related = qs.query.select_related
        if not select_related:
            return

        deferred_fields, is_defer = qs.query.deferred_loading
        if not deferred_fields:
            return

        if select_related is True:
            fk_fields = {f.attname for f in qs.model._meta.concrete_fields if isinstance(f, ForeignKey)}
        else:
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

        if is_defer:
            new_deferred = deferred_fields - fk_fields
            qs.query.deferred_loading = (new_deferred, True)
        else:
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
    """

    _nested_prefetch_lookups: tuple[Any, ...] = ()
    _as_attr_dicts: bool = False

    def _clone(self) -> Self:
        """Clone the queryset, preserving our custom attributes."""
        clone: Self = super()._clone()  # type: ignore[misc]
        clone._nested_prefetch_lookups = self._nested_prefetch_lookups
        clone._as_attr_dicts = self._as_attr_dicts
        return clone

    @overload
    def values_nested(
        self,
        *,
        as_attr_dicts: Literal[False] = ...,
    ) -> QuerySet[_ModelT_co, dict[str, Any]]: ...

    @overload
    def values_nested(
        self,
        *,
        as_attr_dicts: Literal[True],
    ) -> QuerySet[_ModelT_co, AttrDict]: ...

    def values_nested(
        self,
        *,
        as_attr_dicts: bool = False,
    ) -> QuerySet[_ModelT_co, dict[str, Any]] | QuerySet[_ModelT_co, AttrDict]:
        """Return nested dictionaries with related objects included.

        Args:
            as_attr_dicts: If True, return AttrDict instances instead of plain dicts.
                AttrDict supports attribute access (book.title) in addition to
                dict access (book["title"]).

        Returns:
            A QuerySet that yields dict[str, Any] or AttrDict when iterated.

        """
        clone: Self = self._clone()  # type: ignore[assignment]
        clone._iterable_class = NestedValuesIterable
        clone._as_attr_dicts = as_attr_dicts
        clone._nested_prefetch_lookups = clone._prefetch_related_lookups  # type: ignore[attr-defined]
        clone._prefetch_related_lookups = ()  # type: ignore[attr-defined]
        return clone  # type: ignore[return-value]

    def _build_main_queryset(self) -> QuerySet[Any, Any]:
        """Build a fresh queryset for the main query."""
        main_qs = self.model._default_manager.using(self.db).all()
        main_qs.query = self.query.chain()
        main_qs.query.values_select = ()
        return main_qs

    def _fetch_all_prefetched(
        self,
        prefetch_lookups: tuple[Any, ...],
        parent_pks: list[Any],
        main_results: list[dict[str, Any]],
        *,
        container: _ContainerType = dict,
    ) -> dict[str, dict[Any, Any]]:
        """Fetch all prefetched relations."""
        result: dict[str, dict[Any, Any]] = {}
        lookup_map = self._group_prefetch_lookups(prefetch_lookups)

        for attr_name, lookup_info in lookup_map.items():
            lookup = lookup_info["lookup"]
            nested = lookup_info["nested"]

            if lookup_info.get("is_generic_fk"):
                result[attr_name] = self._fetch_generic_fk_values(
                    lookup,
                    parent_pks,
                    main_results,
                    container=container,
                )
            else:
                result[attr_name] = self._fetch_relation_values(
                    lookup,
                    nested,
                    parent_pks,
                    main_results,
                    container=container,
                )

        return result

    def _group_prefetch_lookups(self, prefetch_lookups: tuple[Any, ...]) -> dict[str, dict[str, Any]]:
        """Group prefetch lookups by their top-level attribute name."""
        result: dict[str, dict[str, Any]] = {}

        for lookup in prefetch_lookups:
            if isinstance(lookup, GenericPrefetch):
                attr_name = lookup.to_attr or lookup.prefetch_to
                result[attr_name] = {"lookup": lookup, "nested": [], "is_generic_fk": True}
            elif isinstance(lookup, Prefetch):
                to_attr, _ = lookup.get_current_to_attr(0)
                attr_name = to_attr or lookup.prefetch_to.split("__")[0]
                relation_path = lookup.prefetch_through if to_attr else lookup.prefetch_to

                if attr_name not in result:
                    result[attr_name] = {"lookup": lookup, "nested": []}

                if "__" in relation_path:
                    parts = relation_path.split("__", 1)
                    if parts[0] == attr_name:
                        result[attr_name]["nested"].append(parts[1])
            else:
                attr_name = lookup.split("__")[0]
                relation_path = lookup

                if attr_name not in result:
                    result[attr_name] = {"lookup": lookup, "nested": []}

                if "__" in relation_path:
                    parts = relation_path.split("__", 1)
                    if parts[0] == attr_name:
                        result[attr_name]["nested"].append(parts[1])

        return result

    def _fetch_relation_values(
        self,
        lookup: str | Prefetch[Any],
        nested_relations: list[str],
        parent_pks: list[Any],
        main_results: list[dict[str, Any]],
        *,
        container: _ContainerType = dict,
    ) -> dict[Any, list[dict[str, Any]] | dict[str, Any] | None]:
        """Fetch a related model's data and group by parent PK."""
        relation_name = (
            lookup.prefetch_through.split("__")[0] if isinstance(lookup, Prefetch) else lookup.split("__")[0]
        )

        try:
            field = self.model._meta.get_field(relation_name)
        except FieldDoesNotExist:
            return {}

        custom_qs: QuerySet[Any, Any] | None = (
            lookup.queryset if isinstance(lookup, Prefetch) and lookup.queryset is not None else None
        )
        pk_name = self.model._meta.pk.name

        return self._dispatch_relation_fetch(
            parent_model=self.model,
            field=field,
            nested_relations=nested_relations,
            parent_pks=parent_pks,
            main_results=main_results,
            parent_path="",
            parent_data={r[pk_name]: r for r in main_results},
            custom_qs=custom_qs,
            container=container,
        )

    def _get_select_related_from_queryset(self, qs: QuerySet[Any, Any] | None) -> dict[str, Any]:
        """Get select_related structure from a queryset."""
        if qs is None:
            return {}

        select_related = qs.query.select_related
        if not select_related:
            return {}

        if select_related is True:
            result: dict[str, Any] = {}
            for field in qs.model._meta.concrete_fields:
                if isinstance(field, ForeignKey):
                    result[field.name] = {}
            return result

        result = {}
        self._flatten_select_related_to_paths(select_related, "", result)
        return result

    def _flatten_select_related_to_paths(
        self,
        select_related: dict[str, Any],
        prefix: str,
        result: dict[str, Any],
    ) -> None:
        """Flatten nested select_related dict to path-based dict."""
        for relation_name, nested in select_related.items():
            full_path = f"{prefix}{relation_name}" if prefix else relation_name
            result[full_path] = {}
            if nested:
                self._flatten_select_related_to_paths(nested, f"{full_path}__", result)

    def _fetch_m2m_internal(  # noqa: C901, PLR0912, PLR0913
        self,
        related_model: type[Model],
        accessor: str,
        relation_name: str,
        nested_relations: list[str],
        parent_pks: list[Any],
        custom_qs: QuerySet[Any, Any] | None,
        main_results: list[dict[str, Any]] | None,
        parent_path: str,
        m2m_field: ManyToManyField[Any, Any] | ManyToManyRel,
        *,
        container: _ContainerType = dict,
    ) -> dict[Any, list[dict[str, Any]]]:
        """Fetch M2M data for forward/reverse M2M lookups."""
        related_pk_name = related_model._meta.pk.name

        if custom_qs is not None:
            related_qs = custom_qs.filter(**{f"{accessor}__in": parent_pks})
        else:
            related_qs = related_model._default_manager.filter(**{f"{accessor}__in": parent_pks})

        actual_field = m2m_field.field if isinstance(m2m_field, ManyToManyRel) else m2m_field

        through_model = actual_field.remote_field.through
        assert through_model is not None  # noqa: S101
        source_field_name = (
            actual_field.m2m_reverse_name() if isinstance(m2m_field, ManyToManyRel) else actual_field.m2m_field_name()
        )
        fk = cast("ForeignKey[Any, Any]", through_model._meta.get_field(source_field_name))
        join_table = through_model._meta.db_table
        qn = connections[self.db].ops.quote_name

        extra_select = {
            f"_prefetch_related_val_{f.attname}": f"{qn(join_table)}.{qn(f.column)}" for f in fk.local_related_fields
        }
        related_qs = related_qs.extra(select=extra_select)  # noqa: S610

        # Build directly into container (dict or AttrDict) - no conversion needed
        containers, extra_values = _execute_prefetch(related_qs, self.db, container)
        if not containers:
            return {pk: [] for pk in parent_pks}

        result: dict[Any, list[dict[str, Any]]] = {pk: [] for pk in parent_pks}
        related_data: dict[Any, dict[str, Any]] = {}

        for row_container, extra_vals in zip(containers, extra_values, strict=True):
            source_pk = next(iter(extra_vals.values())) if extra_vals else None
            if source_pk in result:
                result[source_pk].append(row_container)

            related_pk = row_container.get(related_pk_name)
            if nested_relations and related_pk not in related_data:
                related_data[related_pk] = row_container

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
                    container=container,
                )
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
        custom_qs: QuerySet[Any, Any] | None,
        main_results: list[dict[str, Any]] | None,
        parent_path: str,
        *,
        container: _ContainerType = dict,
    ) -> dict[Any, list[dict[str, Any]]]:
        """Fetch reverse FK data."""
        related_pk_name = related_model._meta.pk.name
        fk_field = related_model._meta.get_field(fk_field_name)
        fk_attname = fk_field.attname  # type: ignore[union-attr]

        if custom_qs is not None:
            related_qs = custom_qs.filter(**{f"{fk_field_name}__in": parent_pks})
        else:
            related_qs = related_model._default_manager.filter(**{f"{fk_field_name}__in": parent_pks})

        # Build directly into container (dict or AttrDict)
        related_data = _execute_queryset(related_qs, self.db, container)
        if not related_data:
            return {pk: [] for pk in parent_pks}

        if nested_relations:
            select_related_paths = self._get_select_related_from_queryset(custom_qs)
            remaining_nested = [rel for rel in nested_relations if rel.split("__")[0] not in select_related_paths]
            if remaining_nested:
                related_pks = [r[related_pk_name] for r in related_data]
                nested_dict = {r[related_pk_name]: r for r in related_data}
                full_path = f"{parent_path}{relation_name}" if parent_path else relation_name
                self._add_nested_relations(
                    related_model,
                    nested_dict,
                    remaining_nested,
                    related_pks,
                    main_results,
                    f"{full_path}__",
                    container=container,
                )
                related_data = list(nested_dict.values())

        result: dict[Any, list[dict[str, Any]]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row.get(fk_attname)
            if parent_pk in result:
                # Remove FK attname from output
                if fk_attname in row:
                    del row[fk_attname]
                result[parent_pk].append(row)

        return result

    def _fetch_fk_internal(  # noqa: C901, PLR0912, PLR0913
        self,
        field: ForeignKey[Any, Any],
        nested_relations: list[str],
        parent_pks: list[Any],
        parent_data: dict[Any, dict[str, Any]] | None,
        custom_qs: QuerySet[Any, Any] | None,
        main_results: list[dict[str, Any]] | None,
        parent_path: str,
        parent_model: type[Model] | None = None,
        *,
        container: _ContainerType = dict,
    ) -> dict[Any, dict[str, Any] | None]:
        """Fetch FK data."""
        related_model = cast("type[Model]", field.related_model)
        fk_attname = field.attname
        relation_name = field.name
        related_pk_name = related_model._meta.pk.name

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
            parent_qs = parent_model._default_manager.filter(pk__in=parent_pks)
            pk_name = parent_model._meta.pk.name
            fk_data = {r[pk_name]: r[fk_attname] for r in parent_qs.values(pk_name, fk_attname)}
        else:
            return dict.fromkeys(parent_pks)

        fk_values = list({v for v in fk_data.values() if v is not None})
        if not fk_values:
            return dict.fromkeys(parent_pks)

        has_select_related = parent_data is not None and any(
            relation_name in row and isinstance(row.get(relation_name), dict) for row in parent_data.values()
        )

        if has_select_related and custom_qs is None:
            related_data: dict[Any, dict[str, Any]] = {}
            for row in parent_data.values():  # type: ignore[union-attr]
                nested = row.get(relation_name)
                if isinstance(nested, dict) and nested.get(related_pk_name) is not None:
                    related_data[nested[related_pk_name]] = container(nested)
        else:
            if custom_qs is not None:
                related_qs = custom_qs.filter(pk__in=fk_values)
            else:
                related_qs = related_model._default_manager.filter(pk__in=fk_values)

            results = _execute_queryset(related_qs, self.db, container)
            related_data = {r[related_pk_name]: r for r in results}

        if not related_data:
            return dict.fromkeys(parent_pks)

        if nested_relations:
            full_path = f"{parent_path}{relation_name}__" if parent_path else f"{relation_name}__"
            self._add_nested_relations(
                related_model,
                related_data,
                nested_relations,
                fk_values,
                main_results,
                full_path,
                container=container,
            )

        result: dict[Any, dict[str, Any] | None] = {}
        for parent_pk in parent_pks:
            fk_value = fk_data.get(parent_pk)
            result[parent_pk] = container(related_data[fk_value]) if fk_value in related_data else None

        return result

    def _fetch_generic_relation_internal(  # noqa: PLR0913
        self,
        field: GenericRelation,
        nested_relations: list[str],
        parent_pks: list[Any],
        parent_model: type[Model],
        custom_qs: QuerySet[Any, Any] | None,
        main_results: list[dict[str, Any]] | None,
        parent_path: str,
        *,
        container: _ContainerType = dict,
    ) -> dict[Any, list[dict[str, Any]]]:
        """Fetch GenericRelation data."""
        related_model = cast("type[Model]", field.related_model)
        ct_field_name = field.content_type_field_name
        obj_id_field_name = field.object_id_field_name
        related_pk_name = related_model._meta.pk.name

        parent_ct = ContentType.objects.get_for_model(parent_model)
        filter_kwargs = {ct_field_name: parent_ct, f"{obj_id_field_name}__in": parent_pks}

        if custom_qs is not None:
            related_qs = custom_qs.filter(**filter_kwargs)
        else:
            related_qs = related_model._default_manager.filter(**filter_kwargs)

        related_data = _execute_queryset(related_qs, self.db, container)

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
                container=container,
            )
            related_data = list(nested_dict.values())

        result: dict[Any, list[dict[str, Any]]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row[obj_id_field_name]
            if obj_id_field_name in row:
                del row[obj_id_field_name]
            ct_key = f"{ct_field_name}_id"
            if ct_key in row:
                del row[ct_key]
            result[parent_pk].append(row)

        return result

    def _fetch_generic_fk_values(  # noqa: C901, PLR0912
        self,
        lookup: GenericPrefetch[Any],
        parent_pks: list[Any],
        main_results: list[dict[str, Any]],
        *,
        container: _ContainerType = dict,
    ) -> dict[Any, dict[str, Any] | None]:
        """Fetch GenericForeignKey data using GenericPrefetch."""
        gfk_attr = lookup.prefetch_to
        gfk_descriptor = getattr(self.model, gfk_attr, None)
        if not isinstance(gfk_descriptor, GenericForeignKey):
            return dict.fromkeys(parent_pks)

        ct_field = gfk_descriptor.ct_field
        fk_field = gfk_descriptor.fk_field
        ct_attname = f"{ct_field}_id"
        pk_name = self.model._meta.pk.name

        parent_gfk_info: dict[Any, tuple[Any, Any]] = {}
        for row in main_results:
            parent_pk = row[pk_name]
            parent_gfk_info[parent_pk] = (row.get(ct_attname), row.get(fk_field))

        ct_to_parents: dict[Any, list[tuple[Any, Any]]] = {}
        for parent_pk, (ct_id, obj_id) in parent_gfk_info.items():
            if ct_id is not None and obj_id is not None:
                if ct_id not in ct_to_parents:
                    ct_to_parents[ct_id] = []
                ct_to_parents[ct_id].append((parent_pk, obj_id))

        ct_to_queryset: dict[int, QuerySet[Any, Any]] = {}
        for qs in lookup.querysets:  # type: ignore[attr-defined]
            ct = ContentType.objects.get_for_model(qs.model)
            ct_to_queryset[ct.id] = qs

        result: dict[Any, dict[str, Any] | None] = dict.fromkeys(parent_pks)

        for ct_id, parent_obj_pairs in ct_to_parents.items():
            if ct_id not in ct_to_queryset:
                continue

            qs = ct_to_queryset[ct_id]
            related_model = qs.model
            related_pk_name = related_model._meta.pk.name
            object_ids = [obj_id for _, obj_id in parent_obj_pairs]

            related_qs = qs.filter(pk__in=object_ids)
            results = _execute_queryset(related_qs, self.db, container)
            related_data = {r[related_pk_name]: r for r in results}

            if qs._prefetch_related_lookups:  # type: ignore[attr-defined]
                nested_pks = list(related_data.keys())
                if nested_pks:
                    nested_prefetched = self._fetch_prefetched_for_related(
                        related_model,
                        qs._prefetch_related_lookups,  # type: ignore[attr-defined]
                        nested_pks,
                        list(related_data.values()),
                        container=container,
                    )
                    for pk_val, row_data in related_data.items():
                        for attr, data_by_pk in nested_prefetched.items():
                            row_data[attr] = data_by_pk.get(pk_val, [])

            for parent_pk, obj_id in parent_obj_pairs:
                if obj_id in related_data:
                    result[parent_pk] = container(related_data[obj_id])

        return result

    def _fetch_prefetched_for_related(
        self,
        model: type[Model],
        prefetch_lookups: tuple[Any, ...],
        parent_pks: list[Any],
        main_results: list[dict[str, Any]],
        *,
        container: _ContainerType = dict,
    ) -> dict[str, dict[Any, list[dict[str, Any]] | dict[str, Any] | None]]:
        """Fetch prefetched relations for a related model."""
        temp_qs = NestedValuesQuerySetMixin.__new__(NestedValuesQuerySetMixin)
        temp_qs.model = model
        temp_qs.db = self.db  # type: ignore[misc]
        temp_qs.query = model._default_manager.all().query

        return temp_qs._fetch_all_prefetched(prefetch_lookups, parent_pks, main_results, container=container)

    def _add_nested_relations(  # noqa: PLR0913
        self,
        model: type[Model],
        data: dict[Any, dict[str, Any]],
        nested_relations: list[str],
        parent_pks: list[Any],
        main_results: list[dict[str, Any]] | None = None,
        parent_path: str = "",
        *,
        container: _ContainerType = dict,
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

            nested_data = self._dispatch_relation_fetch(
                parent_model=model,
                field=field,
                nested_relations=further_nested,
                parent_pks=parent_pks,
                main_results=main_results,
                parent_path=parent_path,
                parent_data=data,
                container=container,
            )

            for pk, row in data.items():
                row[rel_name] = nested_data.get(pk, [] if self._is_many_relation(field) else None)

    def _dispatch_relation_fetch(  # noqa: PLR0913
        self,
        parent_model: type[Model],
        field: Any,
        nested_relations: list[str],
        parent_pks: list[Any],
        main_results: list[dict[str, Any]] | None = None,
        parent_path: str = "",
        parent_data: dict[Any, dict[str, Any]] | None = None,
        custom_qs: QuerySet[Any, Any] | None = None,
        *,
        container: _ContainerType = dict,
    ) -> dict[Any, list[dict[str, Any]] | dict[str, Any] | None]:
        """Dispatch to the appropriate fetch method based on field type."""
        if isinstance(field, ManyToManyField):
            return self._fetch_m2m_internal(  # type: ignore[return-value]
                related_model=cast("type[Model]", field.related_model),
                accessor=field.related_query_name(),
                relation_name=field.name,
                nested_relations=nested_relations,
                parent_pks=parent_pks,
                custom_qs=custom_qs,
                main_results=main_results,
                parent_path=parent_path,
                m2m_field=field,
                container=container,
            )
        if isinstance(field, ManyToOneRel):
            return self._fetch_reverse_fk_internal(  # type: ignore[return-value]
                related_model=cast("type[Model]", field.related_model),
                fk_field_name=field.field.name,
                relation_name=field.get_accessor_name() or field.name,
                nested_relations=nested_relations,
                parent_pks=parent_pks,
                custom_qs=custom_qs,
                main_results=main_results,
                parent_path=parent_path,
                container=container,
            )
        if isinstance(field, ForeignKey):
            return self._fetch_fk_internal(  # type: ignore[return-value]
                field=field,
                nested_relations=nested_relations,
                parent_pks=parent_pks,
                parent_data=parent_data,
                custom_qs=custom_qs,
                main_results=main_results,
                parent_path=parent_path,
                parent_model=parent_model,
                container=container,
            )
        if isinstance(field, ManyToManyRel):
            return self._fetch_m2m_internal(  # type: ignore[return-value]
                related_model=cast("type[Model]", field.related_model),
                accessor=field.field.name,
                relation_name=field.get_accessor_name() or field.name,
                nested_relations=nested_relations,
                parent_pks=parent_pks,
                custom_qs=custom_qs,
                main_results=main_results,
                parent_path=parent_path,
                m2m_field=field,
                container=container,
            )
        if isinstance(field, GenericRelation):
            return self._fetch_generic_relation_internal(  # type: ignore[return-value]
                field,
                nested_relations,
                parent_pks,
                parent_model,
                custom_qs,
                main_results,
                parent_path,
                container=container,
            )
        return {}

    def _is_many_relation(self, field: Any) -> bool:
        """Check if a field represents a many-relation."""
        return isinstance(field, ManyToManyField | ManyToManyRel | ManyToOneRel | GenericRelation)


class NestedValuesQuerySet(NestedValuesQuerySetMixin[_ModelT_co], QuerySet[_ModelT_co, _ModelT_co]):
    """QuerySet that adds .values_nested() for nested dictionaries.

    Usage:
        class Book(models.Model):
            objects = NestedValuesQuerySet.as_manager()

        Book.objects.select_related("publisher").prefetch_related("authors").values_nested()
        # Returns: [{'id': 1, 'title': '...', 'publisher': {...}, 'authors': [...]}, ...]

    With attribute access:
        Book.objects.values_nested(as_attr_dicts=True)
        # Returns AttrDict instances: book.title, book.publisher.name work
    """
