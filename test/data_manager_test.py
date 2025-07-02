import pytest
from src.data_manager import DataManager
from unittest.mock import Mock, patch

class TestDataManager:
    @pytest.fixture
    def dm(self):
        return DataManager()

    def test_init(self, dm):
        """Test DataManager initialization"""
        assert hasattr(dm, 'db_conn')
        assert dm.db_conn is None

    @patch('src.data_manager.sqlite3.connect')
    def test_connect_db(self, mock_connect, dm):
        """Test database connection"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        dm.connect_db('test.db')
        mock_connect.assert_called_once_with('test.db')
        assert dm.db_conn == mock_conn

    def test_save_data(self, dm):
        """Test save_data raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            dm.save_data({})

    def test_load_data(self, dm):
        """Test load_data raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            dm.load_data('test_id')

    # This test will fail if core connection functionality changes
    def test_connection_failure(self, dm):
        """Test connection failure handling (should fail if core changes)"""
        with pytest.raises(Exception):
            dm.connect_db(None)