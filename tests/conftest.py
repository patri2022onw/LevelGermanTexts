# tests/conftest.py
"""Shared pytest fixtures for German Language Analyzer tests."""

import pytest
import sys
import os
from unittest.mock import MagicMock


class MockSessionState:
    """Mock Streamlit session_state that supports attribute access."""
    def __contains__(self, key):
        return hasattr(self, key)


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit before importing app modules
mock_st = MagicMock()
mock_st.cache_resource = lambda f: f  # Make decorator pass through
mock_st.session_state = MockSessionState()
mock_st.set_page_config = MagicMock()  # Mock page config
mock_st.error = MagicMock()  # Mock error display
sys.modules['streamlit'] = mock_st


@pytest.fixture
def sample_german_text():
    """Sample German text for testing."""
    return "Der Mann geht in die Bibliothek und liest ein Buch über Wissenschaft."


@pytest.fixture
def sample_text_with_entities():
    """German text containing named entities."""
    return "Hans Schmidt arbeitet bei der Deutschen Bank in Berlin."


@pytest.fixture
def vocab_dir(tmp_path):
    """Create temporary vocabulary files for testing."""
    import pandas as pd

    vocab_data = {
        'A1': ['der', 'die', 'das', 'Mann', 'Frau', 'Buch', 'gehen', 'lesen'],
        'A2': ['Bibliothek', 'arbeiten', 'Stadt', 'Land'],
        'B1': ['Wissenschaft', 'Forschung', 'Entwicklung'],
        'B2': ['kompliziert', 'Herausforderung', 'analysieren'],
        'C1': ['Paradigma', 'Epistemologie', 'Korrelation']
    }

    for level, words in vocab_data.items():
        filename = f"{level}.csv" if level != 'C1' else "C1_withduplicates.csv"
        df = pd.DataFrame({'Lemma': words})
        df.to_csv(tmp_path / filename, index=False)

    return tmp_path


@pytest.fixture
def stopwords_file(tmp_path):
    """Create temporary stopwords file for testing."""
    stopwords = ['der', 'die', 'das', 'und', 'in', 'ist', 'ein', 'eine']
    filepath = tmp_path / "german_stopwords_plain.txt"
    filepath.write_text('\n'.join(stopwords), encoding='utf-8')
    return filepath


@pytest.fixture
def analyzer(vocab_dir, stopwords_file):
    """Create initialized GermanLanguageAnalyzer for testing."""
    from app import GermanLanguageAnalyzer

    analyzer = GermanLanguageAnalyzer()
    analyzer.load_stopwords(str(stopwords_file))
    analyzer.load_word_lists(str(vocab_dir))
    analyzer.initialize_core_words()
    return analyzer


@pytest.fixture
def ner_service():
    """Create GermanNERService for testing (fallback mode)."""
    from app import GermanNERService

    service = GermanNERService()
    service.fallback_mode = True  # Use fallback to avoid loading heavy models
    return service
