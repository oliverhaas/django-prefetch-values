"""Custom QuerySet that adds .values_nested() for nested dictionaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from django.core.exceptions import FieldDoesNotExist
from django.db.models import ForeignKey, ManyToManyField, ManyToManyRel, ManyToOneRel, Prefetch, QuerySet

if TYPE_CHECKING:
    from collections.abc import Iterator


class NestedValuesQuerySet(QuerySet):
    """QuerySet that adds .values_nested() for nested dictionaries.

    This QuerySet adds the values_nested() method that returns nested dictionaries
    with related objects included as dicts (for FK) or lists of dicts (for M2M/reverse FK).

    Usage:
        class Book(models.Model):
            objects = NestedValuesQuerySet.as_manager()

        # Use standard Django patterns:
        Book.objects.only("title").select_related("publisher").prefetch_related("authors").values_nested()
        # Returns: [{'id': 1, 'title': '...', 'publisher': {...}, 'authors': [...]}, ...]
    """

    # Internal flag to track if values_nested() was called
    _values_nested_enabled: bool

    def values_nested(self) -> Self:
        """Return nested dictionaries with related objects included.

        Takes no arguments. Use standard Django methods to control output:
        - .only() to select which fields to include
        - .select_related() for ForeignKey relations (single dict)
        - .prefetch_related() for ManyToMany/reverse FK relations (list of dicts)
        """
        clone = self._clone()
        clone._values_nested_enabled = True
        return clone

    def _clone(self) -> Self:
        """Clone the queryset, preserving our custom attributes."""
        clone: Self = super()._clone()  # type: ignore[assignment]
        if hasattr(self, "_values_nested_enabled"):
            clone._values_nested_enabled = self._values_nested_enabled
        return clone

    def _fetch_all(self) -> None:
        """Override _fetch_all to use our custom values-based fetching."""
        if self._result_cache is None:
            if getattr(self, "_values_nested_enabled", False):
                self._result_cache = self._execute_values_nested()
            else:
                super()._fetch_all()

    def _execute_values_nested(self) -> list[dict[str, Any]]:
        """Execute the query and return nested dictionaries."""
        # Determine which fields to fetch based on .only() / .defer()
        main_fields = self._get_main_fields()
        select_related_fields = self._get_select_related_fields()
        prefetch_lookups = self._prefetch_related_lookups  # type: ignore[attr-defined]

        pk_name = self.model._meta.pk.name

        # Build fields for main query (include pk, main fields, and select_related fields)
        query_fields = self._build_query_fields(main_fields, select_related_fields, pk_name)

        # Execute main query with select_related joins
        main_qs = self._build_main_queryset()
        main_results = list(main_qs.values(*query_fields))

        if not main_results:
            return []

        # Get PKs for prefetch queries
        pk_values = [r[pk_name] for r in main_results]

        # Fetch prefetched relations (pass main_results to avoid extra queries for FK ids)
        prefetched_data = self._fetch_all_prefetched(prefetch_lookups, pk_values, main_results)

        # Build final results
        return self._build_results(
            main_results,
            main_fields,
            select_related_fields,
            prefetched_data,
            pk_name,
        )

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

        Returns a dict mapping relation name to list of fields to fetch.
        Empty list means all fields.
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
        result = {}
        for relation_name in select_related:
            # Get the fields for this relation based on .only()
            relation_fields = self._get_relation_fields_from_only(relation_name)
            result[relation_name] = relation_fields

        return result

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

        # Get FK attnames that are being select_related'd (we don't want both publisher_id and publisher.*)
        fk_attnames_to_skip = set()
        for relation_name in select_related_fields:
            try:
                rel_field = self.model._meta.get_field(relation_name)
                if isinstance(rel_field, ForeignKey):
                    fk_attnames_to_skip.add(rel_field.attname)  # e.g., publisher_id
            except FieldDoesNotExist:
                pass

        # Add main model fields (excluding FKs that have select_related)
        for field in main_fields:
            if field not in fk_attnames_to_skip and "__" not in field and field not in fields:
                fields.append(field)

        # Add select_related fields with double-underscore notation
        for relation_name, relation_fields in select_related_fields.items():
            # Get the related model to know its pk name
            try:
                rel_field = self.model._meta.get_field(relation_name)
            except FieldDoesNotExist:
                continue

            if not isinstance(rel_field, ForeignKey):
                continue

            related_model = rel_field.related_model
            related_pk = related_model._meta.pk.name

            if relation_fields:
                # Specific fields requested via only()
                # Always include pk for the relation
                if related_pk not in relation_fields:
                    fields.append(f"{relation_name}__{related_pk}")
                fields.extend(f"{relation_name}__{rf}" for rf in relation_fields)
            else:
                # All fields from the related model
                fields.extend(f"{relation_name}__{rf.name}" for rf in related_model._meta.concrete_fields)

        return fields

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
            if isinstance(lookup, Prefetch):
                # Get the attribute name (to_attr or top-level relation)
                to_attr, _ = lookup.get_current_to_attr(0)
                attr_name = to_attr or lookup.prefetch_to.split("__")[0]
                relation_path = lookup.prefetch_through if to_attr else lookup.prefetch_to
            else:
                # String lookup
                attr_name = lookup.split("__")[0]
                relation_path = lookup

            # Track nested relations
            if attr_name not in result:
                result[attr_name] = {"lookup": lookup, "nested": []}

            # Check for nested parts (e.g., "books__chapters" -> "chapters" is nested under "books")
            if "__" in relation_path:
                parts = relation_path.split("__", 1)
                if parts[0] == attr_name or (isinstance(lookup, Prefetch) and attr_name == parts[0]):
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
            return self._fetch_m2m_values(field, nested_relations, parent_pks, custom_qs)  # type: ignore[return-value]
        if isinstance(field, ManyToOneRel):
            return self._fetch_reverse_fk_values(field, nested_relations, parent_pks, custom_qs)  # type: ignore[return-value]
        if isinstance(field, ForeignKey):
            return self._fetch_fk_values(field, nested_relations, parent_pks, main_results, custom_qs)  # type: ignore[return-value]
        if isinstance(field, ManyToManyRel):
            return self._fetch_reverse_m2m_values(field, nested_relations, parent_pks, custom_qs)  # type: ignore[return-value]

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

    def _fetch_m2m_values(
        self,
        field: ManyToManyField,
        nested_relations: list[str],
        parent_pks: list[Any],
        custom_qs: QuerySet | None,
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
            self._add_nested_relations(related_model, related_data, nested_relations, list(related_data.keys()))
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
            self._add_nested_relations(related_model, nested_dict, nested_relations, related_pks)
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

        # Get FK values from main_results (avoids extra query)
        pk_name = self.model._meta.pk.name
        fk_data = {r[pk_name]: r.get(fk_attname) for r in main_results}

        fk_values = list({v for v in fk_data.values() if v is not None})
        if not fk_values:
            return dict.fromkeys(parent_pks)

        fetch_fields = self._get_fields_for_relation(related_model, custom_qs)
        related_pk_name = related_model._meta.pk.name
        if related_pk_name not in fetch_fields:
            fetch_fields = [related_pk_name, *fetch_fields]

        # Build queryset
        if custom_qs is not None:
            related_qs = custom_qs.filter(pk__in=fk_values)
        else:
            related_qs = related_model._default_manager.filter(pk__in=fk_values)

        related_data = {r[related_pk_name]: dict(r) for r in related_qs.values(*fetch_fields)}

        # Handle nested relations
        if nested_relations:
            self._add_nested_relations(related_model, related_data, nested_relations, fk_values)

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
            self._add_nested_relations(related_model, related_data, nested_relations, list(related_data.keys()))

        # Group by parent PK
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for through_row in through_data:
            parent_pk = through_row[source_col]
            related_pk = through_row[target_col]
            if related_pk in related_data:
                result[parent_pk].append(dict(related_data[related_pk]))

        return result

    def _build_results(
        self,
        main_results: list[dict],
        main_fields: list[str],
        select_related_fields: dict[str, list[str]],
        prefetched_data: dict[str, dict[Any, list[dict] | dict | None]],
        pk_name: str,
    ) -> list[dict[str, Any]]:
        """Build the final nested dictionaries."""
        result = []

        for row in main_results:
            row_dict: dict[str, Any] = {}
            pk = row[pk_name]

            # Add pk if it was in main_fields
            if pk_name in main_fields:
                row_dict[pk_name] = pk

            # Add main model fields
            for field in main_fields:
                if field in row and field not in select_related_fields and "__" not in field:
                    row_dict[field] = row[field]

            # Build nested dicts for select_related
            for relation_name, relation_fields in select_related_fields.items():
                nested_dict = self._extract_relation_from_row(row, relation_name, relation_fields)
                if nested_dict is not None:
                    row_dict[relation_name] = nested_dict

            # Add prefetched relations
            for attr_name, data_by_pk in prefetched_data.items():
                row_dict[attr_name] = data_by_pk.get(pk, [])

            result.append(row_dict)

        return result

    def _extract_relation_from_row(
        self,
        row: dict,
        relation_name: str,
        relation_fields: list[str],
    ) -> dict | None:
        """Extract a select_related relation's data from a flat row dict."""
        prefix = f"{relation_name}__"
        nested_dict = {}

        # Get related model's pk name
        related_pk_name = None
        try:
            rel_field = self.model._meta.get_field(relation_name)
            if isinstance(rel_field, ForeignKey):
                related_pk_name = rel_field.related_model._meta.pk.name
        except FieldDoesNotExist:
            pass

        for key, value in row.items():
            if key.startswith(prefix):
                field_name = key[len(prefix) :]
                # Always include pk, and check if other fields were requested
                # (empty list means all fields)
                if field_name == related_pk_name or not relation_fields or field_name in relation_fields:
                    nested_dict[field_name] = value

        # Check for NULL FK (related pk is None)
        if related_pk_name and nested_dict.get(related_pk_name) is None:
            return None

        return nested_dict if nested_dict else None

    def _add_nested_relations(
        self,
        model: type,
        data: dict[Any, dict],
        nested_relations: list[str],
        parent_pks: list[Any],
    ) -> None:
        """Add nested relation data to already-fetched data."""
        for nested_rel in nested_relations:
            parts = nested_rel.split("__", 1)
            rel_name = parts[0]
            further_nested = [parts[1]] if len(parts) > 1 else []

            try:
                field = model._meta.get_field(rel_name)  # type: ignore[union-attr]
            except FieldDoesNotExist:
                continue

            nested_data = self._fetch_nested_relation(model, field, rel_name, further_nested, parent_pks)

            for pk, row in data.items():
                row[rel_name] = nested_data.get(pk, [] if self._is_many_relation(field) else None)

    def _fetch_nested_relation(
        self,
        parent_model: type,
        field: Any,
        relation_name: str,
        further_nested: list[str],
        parent_pks: list[Any],
    ) -> dict[Any, list[dict] | dict | None]:
        """Fetch a nested relation for already-fetched data."""
        if isinstance(field, ManyToManyField):
            return self._fetch_nested_m2m(parent_model, field, further_nested, parent_pks)  # type: ignore[return-value]
        if isinstance(field, ManyToOneRel):
            return self._fetch_nested_reverse_fk(parent_model, field, further_nested, parent_pks)  # type: ignore[return-value]
        if isinstance(field, ForeignKey):
            return self._fetch_nested_fk(parent_model, field, further_nested, parent_pks)  # type: ignore[return-value]
        if isinstance(field, ManyToManyRel):
            return self._fetch_nested_reverse_m2m(parent_model, field, further_nested, parent_pks)  # type: ignore[return-value]
        return {}

    def _fetch_nested_m2m(
        self,
        parent_model: type,
        field: ManyToManyField,
        further_nested: list[str],
        parent_pks: list[Any],
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
            self._add_nested_relations(related_model, related_data, further_nested, list(related_data.keys()))

        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for through_row in through_data:
            parent_pk = through_row[source_col]
            related_pk = through_row[target_col]
            if related_pk in related_data:
                result[parent_pk].append(dict(related_data[related_pk]))

        return result

    def _fetch_nested_reverse_fk(
        self,
        parent_model: type,
        field: ManyToOneRel,
        further_nested: list[str],
        parent_pks: list[Any],
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
            self._add_nested_relations(related_model, nested_dict, further_nested, related_pks)
            related_data = list(nested_dict.values())

        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row[fk_field_name]
            row_dict = dict(row)
            row_dict.pop(fk_field_name, None)
            result[parent_pk].append(row_dict)

        return result

    def _fetch_nested_fk(
        self,
        parent_model: type,
        field: ForeignKey,
        further_nested: list[str],
        parent_pks: list[Any],
    ) -> dict[Any, dict | None]:
        """Fetch nested ForeignKey data."""
        related_model = field.related_model
        fk_attr = field.name

        fk_column = f"{fk_attr}_id"
        parent_qs = parent_model._default_manager.filter(pk__in=parent_pks)  # type: ignore[union-attr]
        pk_name = parent_model._meta.pk.name  # type: ignore[union-attr]
        fk_data = {r[pk_name]: r[fk_column] for r in parent_qs.values(pk_name, fk_column)}

        fk_values = list({v for v in fk_data.values() if v is not None})
        if not fk_values:
            return dict.fromkeys(parent_pks)

        fetch_fields = [f.name for f in related_model._meta.concrete_fields]
        related_pk_name = related_model._meta.pk.name

        related_qs = related_model._default_manager.filter(pk__in=fk_values)
        related_data = {r[related_pk_name]: dict(r) for r in related_qs.values(*fetch_fields)}

        if further_nested:
            self._add_nested_relations(related_model, related_data, further_nested, fk_values)

        result: dict[Any, dict | None] = {}
        for parent_pk in parent_pks:
            fk_value = fk_data.get(parent_pk)
            result[parent_pk] = dict(related_data[fk_value]) if fk_value in related_data else None

        return result

    def _fetch_nested_reverse_m2m(
        self,
        parent_model: type,
        field: ManyToManyRel,
        further_nested: list[str],
        parent_pks: list[Any],
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
            self._add_nested_relations(related_model, related_data, further_nested, list(related_data.keys()))

        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for through_row in through_data:
            parent_pk = through_row[source_col]
            related_pk = through_row[target_col]
            if related_pk in related_data:
                result[parent_pk].append(dict(related_data[related_pk]))

        return result

    def _is_many_relation(self, field: Any) -> bool:
        """Check if a field represents a many-relation."""
        return isinstance(field, ManyToManyField | ManyToManyRel | ManyToOneRel)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the queryset."""
        self._fetch_all()
        yield from self._result_cache  # type: ignore[misc]
