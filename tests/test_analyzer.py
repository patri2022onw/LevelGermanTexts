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


class TestUnlistedWords:
    """Words in no vocabulary file must still reach the word list."""

    def test_level_bucket_unlisted_word(self, analyzer):
        """A word in no vocabulary file is bucketed as unlisted at every level."""
        from app import UNKNOWN_LEVEL

        for target in ('A1', 'B1', 'C1'):
            assert analyzer.level_bucket('Dekarbonisierung', target) == UNKNOWN_LEVEL

    def test_level_bucket_at_or_below_target(self, analyzer):
        """A word at or below the target level is not bucketed."""
        assert analyzer.level_bucket('wissenschaft', 'B1') is None
        assert analyzer.level_bucket('buch', 'B1') is None

    def test_level_bucket_above_target(self, analyzer):
        """A listed word above the target keeps its own CEFR level."""
        assert analyzer.level_bucket('kompliziert', 'A1') == 'B2'

    def test_analyze_text_includes_unlisted_words(self, analyzer):
        """Unlisted words appear in the results, even at the highest level."""
        from app import UNKNOWN_LEVEL

        # lowercase to avoid the fallback NER treating them as named entities
        text = "das ist sehr kompliziert und dekarbonisierung ist schwierig"
        result = analyzer.analyze_text(text, 'C1')

        unlisted = [w['lemma'] for w in result['words_above_level'].get(UNKNOWN_LEVEL, [])]
        # simplemma restores canonical German capitalization on the lemma
        assert 'Dekarbonisierung' in unlisted
        # 'kompliziert' is B2, so at target C1 it is not flagged
        assert 'kompliziert' not in unlisted

    def test_analyze_text_skips_numbers_and_single_letters(self, analyzer):
        """Digits and one-letter fragments never become vocabulary entries."""
        result = analyzer.analyze_text("im jahr 2024 z b viel", 'A1')

        lemmas = [lemma for _, lemma in result['all_words']]
        assert '2024' not in lemmas
        assert 'z' not in lemmas
        assert 'b' not in lemmas


class TestWordListBuilding:
    """Tests for create_word_lists output shape."""

    def test_word_list_deduplicates_and_counts(self, analyzer):
        """A repeated word yields one row carrying its occurrence count."""
        from app import create_word_lists

        text = "dekarbonisierung und dekarbonisierung und dekarbonisierung"
        result = analyzer.analyze_text(text, 'B1')
        df = create_word_lists(analyzer, result, 'English', None, 'None')

        rows = df[df['Lemma'] == 'Dekarbonisierung']
        assert len(rows) == 1
        assert rows.iloc[0]['Count'] == 3

    def test_word_list_empty_has_columns(self, analyzer):
        """An empty result still carries the expected columns."""
        from app import create_word_lists

        df = create_word_lists(analyzer, {'words_above_level': {}}, 'English', None, 'None')
        assert df.empty
        assert list(df.columns) == ['German Word', 'Lemma', 'Level', 'Count', 'Translation']
