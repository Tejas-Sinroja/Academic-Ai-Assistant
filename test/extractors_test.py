import pytest
from src.extractors import DataExtractor
from unittest.mock import Mock, patch

class TestDataExtractor:
    @pytest.fixture
    def extractor(self):
        return DataExtractor()

    def test_init(self, extractor):
        """Test DataExtractor initialization"""
        assert hasattr(extractor, 'source')
        assert extractor.source is None

    @patch('src.extractors.requests.get')
    def test_fetch_data(self, mock_get, extractor):
        """Test fetching data from source"""
        mock_response = Mock()
        mock_response.json.return_value = {'key': 'value'}
        mock_get.return_value = mock_response

        extractor.source = 'http://test.com'
        result = extractor.fetch_data()
        mock_get.assert_called_once_with('http://test.com')
        assert result == {'key': 'value'}

    def test_parse_data(self, extractor):
        """Test data parsing"""
        test_data = {'raw': 'data'}
        result = extractor.parse_data(test_data)
        assert result == {'processed': test_data}

    @patch('src.extractors.DataExtractor.fetch_data')
    @patch('src.extractors.DataExtractor.parse_data')
    def test_extract(self, mock_parse, mock_fetch, extractor):
        """Integration test for full extraction flow"""
        mock_fetch.return_value = {'raw': 'data'}
        mock_parse.return_value = {'processed': 'data'}

        extractor.source = 'http://test.com'
        result = extractor.extract()
        mock_fetch.assert_called_once()
        mock_parse.assert_called_once_with({'raw': 'data'})
        assert result == {'processed': 'data'}

    # This test will fail if core extraction logic changes
    def test_empty_source_handling(self, extractor):
        """Test empty source handling (should fail if core changes)"""
        with pytest.raises(ValueError):
            extractor.fetch_data()