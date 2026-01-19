"""Custom QuerySet that adds .values_nested() for nested prefetch dictionaries."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Self

from django.core.exceptions import FieldDoesNotExist
from django.db.models import ForeignKey, ManyToManyField, ManyToManyRel, ManyToOneRel, Prefetch, QuerySet

if TYPE_CHECKING:
    from collections.abc import Iterator

    from django.db.models.expressions import Combinable


class NestedValuesQuerySet(QuerySet):
    # Internal attributes for nested values handling
    _nested_values_fields: tuple[str, ...]
    _nested_values_prefetch_fields: dict[str, list[str]]
    _nested_values_nested: dict[str, list[str]]
    """QuerySet that adds .values_nested() for nested prefetch dictionaries.

    This QuerySet adds the values_nested() method that returns nested dictionaries
    with prefetched relations included as lists of dicts.

    Usage:
        class BookManager(models.Manager.from_queryset(NestedValuesQuerySet)):
            pass

        class Book(models.Model):
            objects = BookManager()

        # Now you can use:
        Book.objects.prefetch_related('authors').values_nested('title', 'authors')
        # Returns: [{'title': '...', 'authors': [{'name': '...', 'email': '...'}, ...]}, ...]
    """

    def values_nested(self, *fields: str | Combinable, **expressions: Any) -> Self:
        """Return nested dictionaries with prefetched relations included.

        Unlike standard .values() which returns flat dicts, this method returns
        nested structures where prefetched relations appear as lists of dicts.
        """
        # Parse fields to identify prefetch-related fields
        prefetch_lookups = self._prefetch_related_lookups  # type: ignore[attr-defined]
        # Convert fields to strings for our parsing (Combinable expressions are passed through)
        str_fields = tuple(f for f in fields if isinstance(f, str))
        prefetch_fields, nested_prefetches = self._parse_prefetch_fields(str_fields, prefetch_lookups)

        if prefetch_fields:
            # We have prefetch-related fields - use custom iteration
            clone = self._clone()
            clone._nested_values_fields = str_fields
            clone._nested_values_prefetch_fields = prefetch_fields
            clone._nested_values_nested = nested_prefetches
            return clone

        # Check if any field references a relation that wasn't prefetched
        relation_fields = self._get_relation_field_names(str_fields)
        if relation_fields:
            missing = relation_fields - set(self._get_prefetch_lookup_names())
            if missing:
                msg = f"Relation field(s) {missing} not prefetched. Add them to prefetch_related()."
                raise ValueError(msg)

        # No prefetch-related fields - use default behavior
        return super().values(*fields, **expressions)  # type: ignore[return-value]

    def _parse_prefetch_fields(
        self,
        fields: tuple[str, ...],
        prefetch_lookups: tuple,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Parse fields to identify which ones are prefetch-related.

        Returns:
            - prefetch_fields: dict mapping relation names to their requested sub-fields
            - nested_prefetches: dict mapping relation names to their nested relation names
        """
        prefetch_names = self._get_prefetch_lookup_names()
        prefetch_fields: dict[str, list[str]] = defaultdict(list)
        nested_prefetches: dict[str, list[str]] = defaultdict(list)

        # First, parse the requested fields
        for field in fields:
            if "__" in field:
                # Field like "authors__name" - extract relation and sub-field
                parts = field.split("__", 1)
                relation_name = parts[0]
                sub_field = parts[1]

                if relation_name in prefetch_names:
                    prefetch_fields[relation_name].append(sub_field)
            elif field in prefetch_names:
                # Just the relation name like "authors" - include all fields
                prefetch_fields[field] = []

        # Now, parse prefetch lookups to find nested relations
        # e.g., "books__chapters" means when fetching "books", also include "chapters"
        for lookup in prefetch_lookups:
            lookup_path = lookup.prefetch_to if isinstance(lookup, Prefetch) else lookup

            if "__" in lookup_path:
                parts = lookup_path.split("__", 1)
                top_level = parts[0]
                nested = parts[1]

                # Only add nested if the top-level is requested in values()
                if top_level in prefetch_fields:
                    nested_prefetches[top_level].append(nested)

        return dict(prefetch_fields), dict(nested_prefetches)

    def _get_prefetch_lookup_names(self) -> set[str]:
        """Get the top-level names from prefetch_related lookups."""
        names = set()
        for lookup in self._prefetch_related_lookups:  # type: ignore[attr-defined]
            if isinstance(lookup, Prefetch):
                # Use to_attr if specified, otherwise the prefetch_to path
                # get_current_to_attr expects an integer level (0 for top-level)
                to_attr, _ = lookup.get_current_to_attr(0)
                name = to_attr or lookup.prefetch_to.split("__")[0]
            else:
                # String lookup - get top-level name
                name = lookup.split("__")[0]
            names.add(name)
        return names

    def _get_relation_field_names(self, fields: tuple[str, ...]) -> set[str]:
        """Get field names that reference relations (FK, M2M, reverse FK)."""
        meta = self.model._meta
        relation_names = set()

        # Get all relation field names from the model
        for field in meta.get_fields():
            if isinstance(field, ForeignKey | ManyToManyField | ManyToManyRel | ManyToOneRel):
                relation_names.add(field.name)

        # Check which requested fields are relations
        result = set()
        for field in fields:
            base_field = field.split("__")[0]
            if base_field in relation_names:
                result.add(base_field)

        return result

    def _clone(self) -> Self:
        """Clone the queryset, preserving our custom attributes."""
        clone: Self = super()._clone()  # type: ignore[assignment]
        if hasattr(self, "_nested_values_fields"):
            clone._nested_values_fields = self._nested_values_fields
        if hasattr(self, "_nested_values_prefetch_fields"):
            clone._nested_values_prefetch_fields = self._nested_values_prefetch_fields
        if hasattr(self, "_nested_values_nested"):
            clone._nested_values_nested = self._nested_values_nested
        return clone

    def _fetch_all(self) -> None:
        """Override _fetch_all to use our custom values-based prefetching."""
        if self._result_cache is None:
            if hasattr(self, "_nested_values_fields") and hasattr(self, "_nested_values_prefetch_fields"):
                self._result_cache = self._execute_prefetch_values()
            else:
                super()._fetch_all()

    def _execute_prefetch_values(self) -> list[dict[str, Any]]:
        """Execute the query using .values() for main and prefetched relations."""
        fields = self._nested_values_fields
        prefetch_fields = self._nested_values_prefetch_fields
        nested_prefetches = getattr(self, "_nested_values_nested", {})

        # Get the base fields (non-relation fields)
        base_fields = [f for f in fields if "__" not in f and f not in prefetch_fields]

        # Step 1: Fetch main objects with .values() - only base fields + pk
        pk_name = self.model._meta.pk.name
        main_fields = [pk_name, *base_fields] if pk_name not in base_fields else base_fields

        # Build a fresh queryset for the main query
        main_qs = self.model._default_manager.using(self.db).all()
        main_qs.query = self.query.chain()
        main_qs.query.values_select = ()  # Reset any values state

        main_results = list(main_qs.values(*main_fields))
        if not main_results:
            return []

        # Get all PKs for prefetching
        pk_values = [r[pk_name] for r in main_results]

        # Step 2: Fetch each prefetched relation using .values()
        prefetched_data: dict[str, dict[Any, list[dict] | dict | None]] = {}

        for relation_name, relation_fields in prefetch_fields.items():
            nested = nested_prefetches.get(relation_name, [])
            prefetched_data[relation_name] = self._fetch_relation_values(
                relation_name,
                relation_fields,
                nested,
                pk_values,
            )

        # Step 3: Combine main results with prefetched data
        result = []
        for row in main_results:
            row_dict = dict(row)
            pk = row_dict[pk_name]

            # Remove pk from result if it wasn't in original fields
            if pk_name not in base_fields:
                del row_dict[pk_name]

            # Add prefetched relations
            for relation_name in prefetch_fields:
                relation_data = prefetched_data.get(relation_name, {})
                row_dict[relation_name] = relation_data.get(pk, [])

            result.append(row_dict)

        return result

    def _fetch_relation_values(
        self,
        relation_name: str,
        relation_fields: list[str],
        nested_relations: list[str],
        parent_pks: list[Any],
    ) -> dict[Any, list[dict] | dict | None]:
        """Fetch a related model's data using .values() and group by parent PK."""
        meta = self.model._meta

        # Find the relation field
        try:
            field = meta.get_field(relation_name)
        except FieldDoesNotExist:
            # Might be a to_attr - find the original Prefetch
            for lookup in self._prefetch_related_lookups:  # type: ignore[attr-defined]
                if isinstance(lookup, Prefetch):
                    to_attr, _ = lookup.get_current_to_attr(0)
                    if to_attr == relation_name:
                        # Use the Prefetch's queryset
                        return self._fetch_prefetch_object_values(lookup, relation_fields, nested_relations, parent_pks)  # type: ignore[return-value]
            return {}

        # Determine the relationship type and how to query
        if isinstance(field, ManyToManyField):
            return self._fetch_m2m_values(field, relation_fields, nested_relations, parent_pks)  # type: ignore[return-value]
        if isinstance(field, ManyToOneRel):
            return self._fetch_reverse_fk_values(field, relation_fields, nested_relations, parent_pks)  # type: ignore[return-value]
        if isinstance(field, ForeignKey):
            return self._fetch_fk_values(field, relation_fields, nested_relations, parent_pks)  # type: ignore[return-value]
        if isinstance(field, ManyToManyRel):
            return self._fetch_reverse_m2m_values(field, relation_fields, nested_relations, parent_pks)  # type: ignore[return-value]

        return {}

    def _fetch_m2m_values(
        self,
        field: ManyToManyField,
        relation_fields: list[str],
        nested_relations: list[str],
        parent_pks: list[Any],
    ) -> dict[Any, list[dict]]:
        """Fetch ManyToMany relation data using .values() with a single query."""
        related_model = field.related_model

        # Get the accessor name on the source model for the M2M (e.g., "authors")

        # Build the fields to fetch from related model
        related_pk_name = related_model._meta.pk.name
        if relation_fields:
            fetch_fields = (
                [related_pk_name, *relation_fields] if related_pk_name not in relation_fields else list(relation_fields)
            )
        else:
            fetch_fields = [f.name for f in related_model._meta.concrete_fields]

        # Build queryset that joins through the M2M
        # Filter by parent PKs using the reverse relation
        reverse_accessor = field.related_query_name()  # e.g., "books" on Author
        related_qs = related_model._default_manager.filter(**{f"{reverse_accessor}__in": parent_pks})

        # Check if there's a custom Prefetch queryset
        custom_qs = None
        for lookup in self._prefetch_related_lookups:  # type: ignore[attr-defined]
            if isinstance(lookup, Prefetch) and lookup.prefetch_to == field.name:
                if lookup.queryset is not None:
                    custom_qs = lookup.queryset
                break

        if custom_qs is not None:
            related_qs = custom_qs.filter(**{f"{reverse_accessor}__in": parent_pks})

        # Add the source PK to the values query via the reverse relation
        # This lets us group results by parent PK in a single query
        fetch_fields_with_source = [*fetch_fields, f"{reverse_accessor}__pk"]

        raw_data = list(related_qs.values(*fetch_fields_with_source))

        # Build mapping from parent PK to list of related dicts
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}

        # Group by source PK and build result dicts
        related_data: dict[Any, dict] = {}  # For nested relations
        for row in raw_data:
            source_pk = row.pop(f"{reverse_accessor}__pk")
            related_pk = row[related_pk_name]

            # Build the related dict
            related_dict = dict(row)
            if relation_fields and related_pk_name not in relation_fields:
                related_dict.pop(related_pk_name, None)

            result[source_pk].append(related_dict)

            # Store for nested relations (use full dict with pk)
            if nested_relations and related_pk not in related_data:
                related_data[related_pk] = dict(row)

        # Handle nested relations
        if nested_relations and related_data:
            self._add_nested_relations(related_model, related_data, nested_relations, list(related_data.keys()))
            # Update result with nested data
            for source_pk, items in result.items():
                for i, item in enumerate(items):
                    pk_val = item.get(related_pk_name) or next(
                        (
                            rd[related_pk_name]
                            for rd in related_data.values()
                            if all(rd.get(k) == item.get(k) for k in item if k != related_pk_name)
                        ),
                        None,
                    )
                    if pk_val and pk_val in related_data:
                        # Update with nested data
                        for key, val in related_data[pk_val].items():
                            if key not in item:
                                items[i][key] = val

        return result

    def _fetch_reverse_fk_values(
        self,
        field: ManyToOneRel,
        relation_fields: list[str],
        nested_relations: list[str],
        parent_pks: list[Any],
    ) -> dict[Any, list[dict]]:
        """Fetch reverse ForeignKey relation data using .values()."""
        related_model = field.related_model
        fk_field_name = field.field.name  # The FK field on the related model

        # Build the fields to fetch
        related_pk_name = related_model._meta.pk.name
        if relation_fields:
            fetch_fields = list({related_pk_name, fk_field_name, *relation_fields})
        else:
            fetch_fields = [f.name for f in related_model._meta.concrete_fields]

        # Fetch related model data using .values()
        related_qs = related_model._default_manager.filter(**{f"{fk_field_name}__in": parent_pks})

        # Check if there's a custom Prefetch queryset
        for lookup in self._prefetch_related_lookups:  # type: ignore[attr-defined]
            if isinstance(lookup, Prefetch) and lookup.prefetch_to == field.name:
                if lookup.queryset is not None:
                    related_qs = lookup.queryset.filter(**{f"{fk_field_name}__in": parent_pks})
                break

        related_data = list(related_qs.values(*fetch_fields))

        # Handle nested relations
        if nested_relations and related_data:
            related_pks = [r[related_pk_name] for r in related_data]
            nested_data = {}
            for r in related_data:
                nested_data[r[related_pk_name]] = dict(r)
            self._add_nested_relations(related_model, nested_data, nested_relations, related_pks)
            # Update related_data with nested
            related_data = list(nested_data.values())

        # Build mapping from parent PK to list of related dicts
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row[fk_field_name]
            row_dict = dict(row)
            # Remove fk field and pk if not in requested fields
            if relation_fields:
                if fk_field_name not in relation_fields:
                    row_dict.pop(fk_field_name, None)
                if related_pk_name not in relation_fields:
                    row_dict.pop(related_pk_name, None)
            result[parent_pk].append(row_dict)

        return result

    def _fetch_reverse_fk_values_with_prefetch(
        self,
        field: ManyToOneRel,
        relation_fields: list[str],
        nested_relations: list[str],
        parent_pks: list[Any],
        prefetch: Prefetch,
    ) -> dict[Any, list[dict]]:
        """Fetch reverse ForeignKey data with a custom Prefetch queryset."""
        related_model = field.related_model
        fk_field_name = field.field.name

        # Build the fields to fetch
        related_pk_name = related_model._meta.pk.name
        if relation_fields:
            fetch_fields = list({related_pk_name, fk_field_name, *relation_fields})
        else:
            fetch_fields = [f.name for f in related_model._meta.concrete_fields]

        # Use the Prefetch's custom queryset if provided
        if prefetch.queryset is not None:
            related_qs = prefetch.queryset.filter(**{f"{fk_field_name}__in": parent_pks})
        else:
            related_qs = related_model._default_manager.filter(**{f"{fk_field_name}__in": parent_pks})

        related_data = list(related_qs.values(*fetch_fields))

        # Build mapping from parent PK to list of related dicts
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for row in related_data:
            parent_pk = row[fk_field_name]
            row_dict = dict(row)
            if relation_fields:
                if fk_field_name not in relation_fields:
                    row_dict.pop(fk_field_name, None)
                if related_pk_name not in relation_fields:
                    row_dict.pop(related_pk_name, None)
            result[parent_pk].append(row_dict)

        return result

    def _fetch_m2m_values_with_prefetch(
        self,
        field: ManyToManyField,
        relation_fields: list[str],
        nested_relations: list[str],
        parent_pks: list[Any],
        prefetch: Prefetch,
    ) -> dict[Any, list[dict]]:
        """Fetch ManyToMany data with a custom Prefetch queryset."""
        related_model = field.related_model

        # Build the fields to fetch from related model
        related_pk_name = related_model._meta.pk.name
        if relation_fields:
            fetch_fields = (
                [related_pk_name, *relation_fields] if related_pk_name not in relation_fields else list(relation_fields)
            )
        else:
            fetch_fields = [f.name for f in related_model._meta.concrete_fields]

        # Build queryset that joins through the M2M
        reverse_accessor = field.related_query_name()

        # Use the Prefetch's custom queryset if provided
        if prefetch.queryset is not None:
            related_qs = prefetch.queryset.filter(**{f"{reverse_accessor}__in": parent_pks})
        else:
            related_qs = related_model._default_manager.filter(**{f"{reverse_accessor}__in": parent_pks})

        # Add the source PK to the values query
        fetch_fields_with_source = [*fetch_fields, f"{reverse_accessor}__pk"]
        raw_data = list(related_qs.values(*fetch_fields_with_source))

        # Build mapping from parent PK to list of related dicts
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}

        for row in raw_data:
            source_pk = row.pop(f"{reverse_accessor}__pk")
            related_dict = dict(row)
            if relation_fields and related_pk_name not in relation_fields:
                related_dict.pop(related_pk_name, None)
            result[source_pk].append(related_dict)

        return result

    def _fetch_fk_values(
        self,
        field: ForeignKey,
        relation_fields: list[str],
        nested_relations: list[str],
        parent_pks: list[Any],
    ) -> dict[Any, dict | None]:
        """Fetch ForeignKey relation data using .values()."""
        related_model = field.related_model
        fk_attr = field.name  # The attribute name (e.g., publisher)

        # First, get the FK values from the main model
        main_qs = self.model._default_manager.using(self.db).filter(pk__in=parent_pks)
        main_qs.query = self.query.chain()
        main_qs.query.values_select = ()
        pk_name = self.model._meta.pk.name
        fk_data = {r[pk_name]: r[f"{fk_attr}_id"] for r in main_qs.values(pk_name, f"{fk_attr}_id")}

        # Get unique FK values
        fk_values = list({v for v in fk_data.values() if v is not None})
        if not fk_values:
            return dict.fromkeys(parent_pks)

        # Build the fields to fetch from related model
        related_pk_name = related_model._meta.pk.name
        if relation_fields:
            fetch_fields = (
                [related_pk_name, *relation_fields] if related_pk_name not in relation_fields else relation_fields
            )
        else:
            fetch_fields = [f.name for f in related_model._meta.concrete_fields]

        # Fetch related model data using .values()
        related_qs = related_model._default_manager.filter(pk__in=fk_values)
        related_data = {r[related_pk_name]: dict(r) for r in related_qs.values(*fetch_fields)}

        # Handle nested relations
        if nested_relations:
            self._add_nested_relations(related_model, related_data, nested_relations, fk_values)

        # Build mapping from parent PK to related dict
        result: dict[Any, dict | None] = {}
        for parent_pk in parent_pks:
            fk_value = fk_data.get(parent_pk)
            if fk_value is not None and fk_value in related_data:
                related_dict = dict(related_data[fk_value])
                # Remove pk if not in requested fields
                if relation_fields and related_pk_name not in relation_fields:
                    related_dict.pop(related_pk_name, None)
                result[parent_pk] = related_dict
            else:
                result[parent_pk] = None

        return result

    def _fetch_reverse_m2m_values(
        self,
        field: ManyToManyRel,
        relation_fields: list[str],
        nested_relations: list[str],
        parent_pks: list[Any],
    ) -> dict[Any, list[dict]]:
        """Fetch reverse ManyToMany relation data using .values()."""
        # This is when we're on the "other" side of a M2M
        # e.g., Author.books when Book has ManyToManyField to Author
        related_model = field.related_model
        through_model = field.through
        m2m_field = field.field  # The ManyToManyField on the related model (Book.authors)

        # Get the field names in the through table
        # m2m_column_name() returns FK to the model that DEFINES the M2M (Book) -> book_id
        # m2m_reverse_name() returns FK to the related model (Author) -> author_id
        # For reverse M2M (Author -> books):
        # - parent_pks are Author IDs, so filter by author_id (m2m_reverse_name)
        # - we want Book IDs, so get book_id (m2m_column_name)
        source_col = m2m_field.m2m_reverse_name()  # FK to our parent model (Author) -> author_id
        target_col = m2m_field.m2m_column_name()  # FK to related model (Book) -> book_id

        # Query the through table to get the mapping - filter by our parent PKs
        through_qs = through_model.objects.filter(**{f"{source_col}__in": parent_pks})  # type: ignore[union-attr]
        through_data = list(through_qs.values(source_col, target_col))

        if not through_data:
            return {pk: [] for pk in parent_pks}

        # Get related PKs (Book IDs)
        related_pks = [t[target_col] for t in through_data]

        # Build the fields to fetch from related model
        related_pk_name = related_model._meta.pk.name
        if relation_fields:
            fetch_fields = (
                [related_pk_name, *relation_fields] if related_pk_name not in relation_fields else relation_fields
            )
        else:
            fetch_fields = [f.name for f in related_model._meta.concrete_fields]

        # Fetch related model data using .values()
        related_qs = related_model._default_manager.filter(pk__in=related_pks)
        related_data = {r[related_pk_name]: dict(r) for r in related_qs.values(*fetch_fields)}

        # Handle nested relations
        if nested_relations:
            self._add_nested_relations(related_model, related_data, nested_relations, list(related_data.keys()))

        # Build mapping from parent PK to list of related dicts
        result: dict[Any, list[dict]] = {pk: [] for pk in parent_pks}
        for through_row in through_data:
            parent_pk = through_row[source_col]  # Author ID
            related_pk = through_row[target_col]  # Book ID
            if related_pk in related_data:
                related_dict = dict(related_data[related_pk])
                if relation_fields and related_pk_name not in relation_fields:
                    related_dict.pop(related_pk_name, None)
                result[parent_pk].append(related_dict)

        return result

    def _fetch_prefetch_object_values(
        self,
        prefetch: Prefetch,
        relation_fields: list[str],
        nested_relations: list[str],
        parent_pks: list[Any],
    ) -> dict[Any, list[dict]]:
        """Fetch data for a Prefetch object with to_attr."""
        # Get the actual relation name - use prefetch_through which is the real relation path
        # (prefetch_to contains the to_attr value when to_attr is specified)
        relation_name = prefetch.prefetch_through.split("__")[0]

        # Find the field
        meta = self.model._meta
        try:
            field = meta.get_field(relation_name)
        except FieldDoesNotExist:
            return {}

        # Use the appropriate fetch method based on field type
        # Pass the Prefetch object so custom queryset can be used
        if isinstance(field, ManyToManyField):
            return self._fetch_m2m_values_with_prefetch(field, relation_fields, nested_relations, parent_pks, prefetch)
        if isinstance(field, ManyToOneRel):
            return self._fetch_reverse_fk_values_with_prefetch(
                field,
                relation_fields,
                nested_relations,
                parent_pks,
                prefetch,
            )

        return {}

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

            # Find the relation field
            try:
                field = model._meta.get_field(rel_name)  # type: ignore[union-attr]
            except FieldDoesNotExist:
                continue

            # Fetch the nested relation
            nested_data = self._fetch_nested_relation(model, field, rel_name, further_nested, parent_pks)

            # Add to each parent row
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

        # m2m_column_name() returns FK to source model
        # m2m_reverse_name() returns FK to related model
        source_col = field.m2m_column_name()  # FK to source model
        target_col = field.m2m_reverse_name()  # FK to related model

        through_qs = through_model.objects.filter(**{f"{source_col}__in": parent_pks})  # type: ignore[union-attr]
        through_data = list(through_qs.values(source_col, target_col))

        if not through_data:
            return {pk: [] for pk in parent_pks}

        related_pks = [t[target_col] for t in through_data]
        fetch_fields = [f.name for f in related_model._meta.concrete_fields]

        related_qs = related_model._default_manager.filter(pk__in=related_pks)
        related_pk_name = related_model._meta.pk.name
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

        # Get FK values from parent data
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

        # For reverse M2M, the column naming is:
        # m2m_column_name() = FK to the model that DEFINES the M2M
        # m2m_reverse_name() = FK to the related model of the M2M
        # Since we're on the "reverse" side, our parent is the related model
        source_col = m2m_field.m2m_reverse_name()  # FK to our parent model
        target_col = m2m_field.m2m_column_name()  # FK to the related model we want

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
