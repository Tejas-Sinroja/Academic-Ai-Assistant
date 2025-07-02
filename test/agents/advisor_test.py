import pytest
from src.agents.advisor import Advisor
from unittest.mock import Mock, patch

class TestAdvisor:
    @pytest.fixture
    def advisor(self):
        return Advisor()

    def test_init(self, advisor):
        """Test Advisor initialization"""
        assert hasattr(advisor, 'knowledge_base')
        assert isinstance(advisor.knowledge_base, dict)

    @patch('src.agents.advisor.LLMHandler')
    def test_get_advice(self, mock_llm, advisor):
        """Test getting advice from advisor"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Test advice"
        mock_llm.return_value = mock_llm_instance

        result = advisor.get_advice("test query")
        mock_llm_instance.generate_response.assert_called_once()
        assert result == "Test advice"

    def test_update_knowledge(self, advisor):
        """Test updating knowledge base"""
        test_data = {"key": "value"}
        advisor.update_knowledge(test_data)
        assert advisor.knowledge_base == test_data

    # This test will fail if core advice generation changes
    def test_empty_query_handling(self, advisor):
        """Test empty query handling (should fail if core changes)"""
        with pytest.raises(ValueError):
            advisor.get_advice("")