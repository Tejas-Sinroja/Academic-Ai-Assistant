import pytest
from src.agents.notewriter import NoteWriter
from unittest.mock import Mock, patch

class TestNoteWriter:
    @pytest.fixture
    def notewriter(self):
        return NoteWriter()

    def test_init(self, notewriter):
        """Test NoteWriter initialization"""
        assert hasattr(notewriter, 'template')
        assert isinstance(notewriter.template, str)

    @patch('src.agents.notewriter.LLMHandler')
    def test_generate_note(self, mock_llm, notewriter):
        """Test note generation"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Formatted note content"
        mock_llm.return_value = mock_llm_instance

        result = notewriter.generate_note("raw content")
        mock_llm_instance.generate_response.assert_called_once()
        assert "Formatted note content" in result

    def test_format_note(self, notewriter):
        """Test note formatting"""
        test_content = "test\ncontent"
        result = notewriter.format_note(test_content)
        assert "test" in result
        assert "content" in result
        assert "\n" not in result

    # This test will fail if core note generation changes
    def test_empty_content_handling(self, notewriter):
        """Test empty content handling (should fail if core changes)"""
        with pytest.raises(ValueError):
            notewriter.generate_note("")