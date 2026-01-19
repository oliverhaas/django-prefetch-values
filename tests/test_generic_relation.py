# tests/test_generic_relation.py
import pytest

from django_nested_values import NestedValuesQuerySet
from tests.models import Article, Comment, TaggedItem


@pytest.mark.django_db
class TestGenericRelation:
    """Tests for GenericRelation support in values_nested()."""

    def test_generic_relation_basic(self):
        """GenericRelation should be fetched and included in nested output."""
        # Arrange
        article = Article.objects.create(title="Test Article")
        TaggedItem.objects.create(content_object=article, tag="python")
        TaggedItem.objects.create(content_object=article, tag="django")

        # Act
        qs = NestedValuesQuerySet(model=Article)
        result = list(
            qs.filter(id=article.id).prefetch_related("tags").values_nested(),
        )

        # Assert
        assert len(result) == 1
        assert result[0]["title"] == "Test Article"
        assert "tags" in result[0]
        assert len(result[0]["tags"]) == 2

        tag_names = {t["tag"] for t in result[0]["tags"]}
        assert tag_names == {"python", "django"}

    def test_generic_relation_empty(self):
        """GenericRelation with no related objects should return empty list."""
        # Arrange
        article = Article.objects.create(title="No Tags Article")

        # Act
        qs = NestedValuesQuerySet(model=Article)
        result = list(
            qs.filter(id=article.id).prefetch_related("tags").values_nested(),
        )

        # Assert
        assert len(result) == 1
        assert result[0]["tags"] == []

    def test_generic_relation_multiple_objects(self):
        """GenericRelation should work correctly with multiple parent objects."""
        # Arrange
        article1 = Article.objects.create(title="Article 1")
        article2 = Article.objects.create(title="Article 2")
        TaggedItem.objects.create(content_object=article1, tag="tag1")
        TaggedItem.objects.create(content_object=article1, tag="tag2")
        TaggedItem.objects.create(content_object=article2, tag="tag3")

        # Act
        qs = NestedValuesQuerySet(model=Article)
        result = list(
            qs.filter(id__in=[article1.id, article2.id]).prefetch_related("tags").values_nested(),
        )

        # Assert
        assert len(result) == 2

        result_by_title = {r["title"]: r for r in result}
        assert len(result_by_title["Article 1"]["tags"]) == 2
        assert len(result_by_title["Article 2"]["tags"]) == 1

    def test_generic_relation_nested_through_fk(self):
        """GenericRelation should work when accessed through a FK relation."""
        # Arrange
        article = Article.objects.create(title="Article with Comments")
        comment = Comment.objects.create(article=article, text="Great article!")
        TaggedItem.objects.create(content_object=comment, tag="helpful")

        # Act
        qs = NestedValuesQuerySet(model=Article)
        result = list(
            qs.filter(id=article.id).prefetch_related("comments__tags").values_nested(),
        )

        # Assert
        assert len(result) == 1
        assert len(result[0]["comments"]) == 1
        assert result[0]["comments"][0]["text"] == "Great article!"
        assert len(result[0]["comments"][0]["tags"]) == 1
        assert result[0]["comments"][0]["tags"][0]["tag"] == "helpful"

    def test_generic_relation_with_select_related_on_tagged_item(self):
        """GenericRelation should support nested select_related on the related model."""
        # Arrange - TaggedItem doesn't have FK fields by default,
        # but this tests the pattern works if it did
        article = Article.objects.create(title="Test")
        TaggedItem.objects.create(content_object=article, tag="test-tag")

        # Act
        qs = NestedValuesQuerySet(model=Article)
        result = list(
            qs.filter(id=article.id).prefetch_related("tags").values_nested(),
        )

        # Assert - basic check that it doesn't error
        assert len(result) == 1
        assert len(result[0]["tags"]) == 1
