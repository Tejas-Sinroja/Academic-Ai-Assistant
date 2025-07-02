import pytest
from src.agents.planner import Planner
from unittest.mock import Mock, patch

class TestPlanner:
    @pytest.fixture
    def planner(self):
        return Planner()

    def test_init(self, planner):
        """Test Planner initialization"""
        assert hasattr(planner, 'task_queue')
        assert isinstance(planner.task_queue, list)

    @patch('src.agents.planner.LLMHandler')
    def test_create_plan(self, mock_llm, planner):
        """Test plan creation"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Step 1\nStep 2\nStep 3"
        mock_llm.return_value = mock_llm_instance

        result = planner.create_plan("objective")
        mock_llm_instance.generate_response.assert_called_once()
        assert len(result) == 3
        assert "Step 1" in result

    def test_add_task(self, planner):
        """Test adding tasks to queue"""
        planner.add_task("test_task")
        assert len(planner.task_queue) == 1
        assert planner.task_queue[0] == "test_task"

    # This test will fail if core planning logic changes
    def test_empty_objective_handling(self, planner):
        """Test empty objective handling (should fail if core changes)"""
        with pytest.raises(ValueError):
            planner.create_plan("")