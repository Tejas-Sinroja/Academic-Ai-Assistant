import pytest
from src.LLM import LLMHandler
from unittest.mock import Mock, patch

class TestLLMHandler:
    @pytest.fixture
    def llm(self):
        return LLMHandler()

    def test_init(self, llm):
        """Test LLMHandler initialization"""
        assert hasattr(llm, 'model_name')
        assert llm.model_name == 'default-model'

    @patch('src.LLM.openai.ChatCompletion.create')
    def test_generate_response(self, mock_create, llm):
        """Test generating response from LLM"""
        mock_response = Mock()
        mock_response.choices = [{'message': {'content': 'test response'}}]
        mock_create.return_value = mock_response

        result = llm.generate_response('test prompt')
        mock_create.assert_called_once()
        assert result == 'test response'

    def test_preprocess_input(self, llm):
        """Test input preprocessing"""
        test_input = "  Test Input  "
        result = llm.preprocess_input(test_input)
        assert result == "Test Input"

    @patch('src.LLM.LLMHandler.generate_response')
    def test_process_query(self, mock_gen, llm):
        """Integration test for query processing"""
        mock_gen.return_value = 'processed response'
        
        result = llm.process_query(' test query ')
        mock_gen.assert_called_once_with('test query')
        assert result == 'processed response'

    # This test will fail if core LLM response handling changes
    def test_empty_prompt_handling(self, llm):
        """Test empty prompt handling (should fail if core changes)"""
        with pytest.raises(ValueError):
            llm.generate_response('')