# tests/test_translation.py
"""Tests for translation service."""

import pytest
from unittest.mock import MagicMock, patch


class TestClaudeTranslationProvider:
    """Tests for Claude translation provider."""

    @pytest.fixture
    def mock_claude_client(self):
        """Create a mock Claude client."""
        client = MagicMock()
        response = MagicMock()
        response.content = [MagicMock(text="book")]
        client.messages.create.return_value = response
        return client

    @pytest.fixture
    def provider(self, mock_claude_client):
        """Create ClaudeTranslationProvider with mocked client."""
        from translation_service import ClaudeTranslationProvider
        return ClaudeTranslationProvider(mock_claude_client)

    def test_translate_single_word(self, provider, mock_claude_client):
        """Test translating a single word."""
        result = provider.translate("Buch", "German", "English")

        assert result == "book"
        mock_claude_client.messages.create.assert_called_once()

    def test_translate_uses_correct_model(self, provider, mock_claude_client):
        """Test that translation uses the correct model."""
        provider.translate("Buch")

        call_kwargs = mock_claude_client.messages.create.call_args[1]
        assert call_kwargs['model'] == "claude-sonnet-4-20250514"

    def test_translate_error_returns_original(self, provider, mock_claude_client):
        """Test that translation errors return the original text."""
        mock_claude_client.messages.create.side_effect = Exception("API Error")

        result = provider.translate("Buch")
        assert result == "Buch"

    def test_translate_batch_empty_list(self, provider):
        """Test batch translation with empty list."""
        result = provider.translate_batch([])
        assert result == {}

    def test_translate_batch_returns_dict(self, provider, mock_claude_client):
        """Test batch translation returns dictionary."""
        response = MagicMock()
        response.content = [MagicMock(text="1. book\n2. house")]
        mock_claude_client.messages.create.return_value = response

        result = provider.translate_batch(["Buch", "Haus"])

        assert isinstance(result, dict)
        assert len(result) == 2


class TestGeminiTranslationProvider:
    """Tests for Gemini translation provider."""

    @pytest.fixture
    def mock_gemini_model(self):
        """Create a mock Gemini model."""
        model = MagicMock()
        response = MagicMock()
        response.text = "book"
        model.generate_content.return_value = response
        return model

    @pytest.fixture
    def provider(self, mock_gemini_model):
        """Create GeminiTranslationProvider with mocked model."""
        from translation_service import GeminiTranslationProvider
        return GeminiTranslationProvider(mock_gemini_model)

    def test_translate_single_word(self, provider, mock_gemini_model):
        """Test translating a single word."""
        result = provider.translate("Buch", "German", "English")

        assert result == "book"
        mock_gemini_model.generate_content.assert_called_once()

    def test_translate_error_returns_original(self, provider, mock_gemini_model):
        """Test that translation errors return the original text."""
        mock_gemini_model.generate_content.side_effect = Exception("API Error")

        result = provider.translate("Buch")
        assert result == "Buch"


class TestMockTranslationProvider:
    """Tests for mock translation provider."""

    @pytest.fixture
    def provider(self):
        """Create MockTranslationProvider."""
        from translation_service import MockTranslationProvider
        return MockTranslationProvider()

    def test_translate_known_word(self, provider):
        """Test translating a known word."""
        result = provider.translate("Buch", "German", "English")
        assert result == "book"

    def test_translate_unknown_word(self, provider):
        """Test translating an unknown word returns formatted mock."""
        result = provider.translate("unbekannt", "German", "English")
        assert "(EN)" in result

    def test_translate_batch(self, provider):
        """Test batch translation."""
        result = provider.translate_batch(["Buch", "Haus"], "German", "English")
        assert result["Buch"] == "book"
        assert result["Haus"] == "house"


class TestTranslationService:
    """Tests for main TranslationService class."""

    @pytest.fixture
    def service(self):
        """Create TranslationService with mock provider."""
        from translation_service import TranslationService
        return TranslationService(ai_service=None, ai_model="None")

    def test_uses_mock_provider_when_no_ai(self, service):
        """Test that mock provider is used when no AI configured."""
        from translation_service import MockTranslationProvider
        assert isinstance(service.provider, MockTranslationProvider)

    def test_translate_word_caches_result(self, service):
        """Test that translations are cached."""
        # First call
        result1 = service.translate_word("Buch", "English")
        # Second call should use cache
        result2 = service.translate_word("Buch", "English")

        assert result1 == result2
