# tests/test_ner.py
"""Tests for GermanNERService class."""

import pytest


class TestGermanNERService:
    """Tests for the NER service."""

    def test_fallback_ner_finds_capitalized_words(self, ner_service):
        """Test fallback NER detects capitalized words as potential entities."""
        text = "Hans arbeitet in Berlin."
        entities = ner_service._fallback_ner(text)

        # Should find proper nouns
        assert 'hans' in entities or 'berlin' in entities

    def test_fallback_ner_skips_common_nouns(self, ner_service):
        """Test fallback NER skips common German nouns."""
        text = "Das Haus ist groß."
        entities = ner_service._fallback_ner(text)

        # 'Haus' is a common noun, should be filtered
        assert 'haus' not in entities

    def test_fallback_ner_skips_months(self, ner_service):
        """Test fallback NER skips month names."""
        text = "Im Januar ist es kalt."
        entities = ner_service._fallback_ner(text)

        assert 'januar' not in entities

    def test_fallback_ner_skips_days(self, ner_service):
        """Test fallback NER skips day names."""
        text = "Am Montag gehe ich arbeiten."
        entities = ner_service._fallback_ner(text)

        assert 'montag' not in entities

    def test_is_entity_true(self, ner_service):
        """Test is_entity returns True for known entities."""
        entities = {'hans', 'berlin', 'schmidt'}
        assert ner_service.is_entity('Hans', entities) is True
        assert ner_service.is_entity('BERLIN', entities) is True

    def test_is_entity_false(self, ner_service):
        """Test is_entity returns False for non-entities."""
        entities = {'hans', 'berlin'}
        assert ner_service.is_entity('Buch', entities) is False
        assert ner_service.is_entity('arbeiten', entities) is False

    def test_extract_entities_fallback_mode(self, ner_service):
        """Test extract_entities uses fallback when in fallback mode."""
        text = "Maria wohnt in München."
        entities = ner_service.extract_entities(text)

        # Should return a set
        assert isinstance(entities, set)


class TestNEREdgeCases:
    """Edge case tests for NER service."""

    def test_empty_text(self, ner_service):
        """Test NER handles empty text."""
        entities = ner_service.extract_entities("")
        assert entities == set()

    def test_no_entities_text(self, ner_service):
        """Test NER handles text without entities."""
        text = "ich gehe nach hause"  # All lowercase
        entities = ner_service._fallback_ner(text)
        assert len(entities) == 0

    def test_short_tokens_filtered(self, ner_service):
        """Test that very short tokens are handled."""
        text = "A B C sind Buchstaben."
        entities = ner_service._fallback_ner(text)
        # Single letters should not be entities
        assert 'a' not in entities
        assert 'b' not in entities
