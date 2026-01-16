# tests/test_analyzer.py
"""Tests for GermanLanguageAnalyzer class."""

import pytest


class TestGermanLanguageAnalyzer:
    """Tests for the main analyzer class."""

    def test_load_stopwords(self, analyzer):
        """Test that stopwords are loaded correctly."""
        assert len(analyzer.stopwords) > 0
        assert 'der' in analyzer.stopwords
        assert 'und' in analyzer.stopwords

    def test_load_word_lists(self, analyzer):
        """Test that vocabulary is loaded correctly."""
        assert len(analyzer.word_levels) > 0
        # Check some words are at expected levels
        assert analyzer.word_levels.get('wissenschaft') == 'B1'
        assert analyzer.word_levels.get('bibliothek') == 'A2'

    def test_initialize_core_words(self, analyzer):
        """Test that core words are initialized."""
        assert len(analyzer.core_words) > 0
        assert 'ich' in analyzer.core_words
        assert 'und' in analyzer.core_words

    def test_simple_tokenize(self, analyzer):
        """Test basic tokenization."""
        text = "Der Mann liest."
        tokens = analyzer.simple_tokenize(text)
        assert 'Der' in tokens
        assert 'Mann' in tokens
        assert 'liest' in tokens
        assert '.' in tokens

    def test_get_word_level_known_word(self, analyzer):
        """Test getting level for a known word."""
        level = analyzer.get_word_level('wissenschaft')
        assert level == 'B1'

    def test_get_word_level_unknown_word(self, analyzer):
        """Test getting level for an unknown word."""
        level = analyzer.get_word_level('xyzabc')
        assert level is None

    def test_is_above_level_true(self, analyzer):
        """Test is_above_level returns True when word is above target."""
        assert analyzer.is_above_level('B1', 'A2') is True
        assert analyzer.is_above_level('C1', 'B2') is True

    def test_is_above_level_false(self, analyzer):
        """Test is_above_level returns False when word is at or below target."""
        assert analyzer.is_above_level('A1', 'A2') is False
        assert analyzer.is_above_level('B1', 'B1') is False

    def test_is_above_level_invalid(self, analyzer):
        """Test is_above_level handles invalid levels."""
        assert analyzer.is_above_level('X1', 'A1') is False
        assert analyzer.is_above_level('A1', 'X1') is False

    def test_analyze_text_basic(self, analyzer, sample_german_text):
        """Test basic text analysis."""
        result = analyzer.analyze_text(sample_german_text, 'A1')

        assert 'words_above_level' in result
        assert 'all_words' in result
        assert 'total_words' in result
        assert 'tokens' in result

    def test_analyze_text_finds_above_level_words(self, analyzer):
        """Test that analyzer finds words above target level."""
        # Use lowercase 'kompliziert' to avoid NER filtering (German capitalizes nouns)
        text = "Das ist sehr kompliziert."
        result = analyzer.analyze_text(text, 'A1')

        # 'kompliziert' is B2, should be found when target is A1
        above_level = result['words_above_level']
        all_lemmas = []
        for level_words in above_level.values():
            all_lemmas.extend([w['lemma'].lower() for w in level_words])

        assert 'kompliziert' in all_lemmas

    def test_analyze_text_skips_stopwords(self, analyzer):
        """Test that stopwords are skipped."""
        text = "Der und die"
        result = analyzer.analyze_text(text, 'A1')

        # All these should be skipped as stopwords/core words
        assert result['total_words'] == 0


class TestLevelOrdering:
    """Tests for CEFR level ordering logic."""

    def test_level_order_a1_lowest(self, analyzer):
        """Test A1 is the lowest level."""
        assert analyzer.is_above_level('A2', 'A1') is True
        assert analyzer.is_above_level('B1', 'A1') is True
        assert analyzer.is_above_level('C1', 'A1') is True

    def test_level_order_c1_highest(self, analyzer):
        """Test C1 is the highest level."""
        assert analyzer.is_above_level('C1', 'B2') is True
        assert analyzer.is_above_level('C1', 'B1') is True
        assert analyzer.is_above_level('C1', 'A1') is True

    def test_level_order_same_level(self, analyzer):
        """Test same level is not above."""
        for level in ['A1', 'A2', 'B1', 'B2', 'C1']:
            assert analyzer.is_above_level(level, level) is False
