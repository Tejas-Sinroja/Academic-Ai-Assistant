import pytest
from src.agents.coordinator import Coordinator
from unittest.mock import Mock, patch

class TestCoordinator:
    @pytest.fixture
    def coordinator(self):
        return Coordinator()

    def test_init(self, coordinator):
        """Test Coordinator initialization"""
        assert hasattr(coordinator, 'agents')
        assert isinstance(coordinator.agents, dict)

    @patch('src.agents.coordinator.Advisor')
    @patch('src.agents.coordinator.Planner')
    @patch('src.agents.coordinator.NoteWriter')
    def test_register_agents(self, mock_nw, mock_planner, mock_advisor, coordinator):
        """Test agent registration"""
        coordinator.register_agents()
        mock_advisor.assert_called_once()
        mock_planner.assert_called_once()
        mock_nw.assert_called_once()
        assert len(coordinator.agents) == 3

    @patch.object(Coordinator, 'register_agents')
    def test_delegate_task(self, mock_register, coordinator):
        """Test task delegation"""
        mock_agent = Mock()
        mock_agent.process_task.return_value = "task completed"
        coordinator.agents['test_agent'] = mock_agent

        result = coordinator.delegate_task('test_agent', 'test_task')
        mock_agent.process_task.assert_called_once_with('test_task')
        assert result == "task completed"

    # This test will fail if core delegation logic changes
    def test_invalid_agent_handling(self, coordinator):
        """Test invalid agent handling (should fail if core changes)"""
        with pytest.raises(ValueError):
            coordinator.delegate_task('invalid_agent', 'task')