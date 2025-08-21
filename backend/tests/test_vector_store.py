import pytest
from unittest.mock import Mock, patch
from vector_store import VectorStore, SearchResults


class TestVectorStore:
    """Test suite for VectorStore validation and error handling"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a VectorStore with mocked ChromaDB"""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            store = VectorStore("./test_db", "all-MiniLM-L6-v2", max_results=5)
            
            # Mock the collections
            store.course_catalog = Mock()
            store.course_content = Mock()
            
            return store

    def test_search_with_valid_limit(self, mock_vector_store):
        """Test search with valid limit parameter"""
        # Mock successful ChromaDB response
        mock_vector_store.course_content.query.return_value = {
            'documents': [['Test document']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        result = mock_vector_store.search("test query", limit=3)
        
        assert not result.error
        assert len(result.documents) == 1
        mock_vector_store.course_content.query.assert_called_once()
        
        # Verify limit was used
        call_args = mock_vector_store.course_content.query.call_args[1]
        assert call_args['n_results'] == 3

    def test_search_with_invalid_limit_zero(self, mock_vector_store):
        """Test search with invalid limit of 0"""
        result = mock_vector_store.search("test query", limit=0)
        
        assert result.error == "Invalid search limit: 0. Must be >= 1"
        assert result.is_empty()
        # ChromaDB should not be called
        mock_vector_store.course_content.query.assert_not_called()

    def test_search_with_invalid_limit_negative(self, mock_vector_store):
        """Test search with negative limit"""
        result = mock_vector_store.search("test query", limit=-5)
        
        assert result.error == "Invalid search limit: -5. Must be >= 1"
        assert result.is_empty()
        mock_vector_store.course_content.query.assert_not_called()

    def test_search_with_invalid_max_results_config(self):
        """Test that VectorStore with invalid max_results from config fails validation"""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            # Create store with invalid max_results
            store = VectorStore("./test_db", "all-MiniLM-L6-v2", max_results=0)
            
            # Mock collections
            store.course_catalog = Mock()
            store.course_content = Mock()
            
            # When search is called without explicit limit, should use max_results=0 and fail
            result = store.search("test query")
            
            assert result.error == "Invalid search limit: 0. Must be >= 1"
            assert result.is_empty()

    def test_search_uses_default_max_results_when_no_limit_provided(self, mock_vector_store):
        """Test that search uses configured max_results when no limit provided"""
        # Mock successful response
        mock_vector_store.course_content.query.return_value = {
            'documents': [['Test document']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        result = mock_vector_store.search("test query")  # No limit provided
        
        assert not result.error
        mock_vector_store.course_content.query.assert_called_once()
        
        # Verify default max_results was used
        call_args = mock_vector_store.course_content.query.call_args[1]
        assert call_args['n_results'] == 5  # Default max_results from fixture

    def test_search_with_chromadb_exception(self, mock_vector_store):
        """Test handling of ChromaDB exceptions"""
        # Mock ChromaDB to raise exception
        mock_vector_store.course_content.query.side_effect = Exception("ChromaDB connection failed")
        
        result = mock_vector_store.search("test query")
        
        assert result.error == "Search error: ChromaDB connection failed"
        assert result.is_empty()

    def test_search_results_empty_classmethod(self):
        """Test SearchResults.empty class method"""
        error_msg = "Test error message"
        result = SearchResults.empty(error_msg)
        
        assert result.error == error_msg
        assert result.is_empty()
        assert result.documents == []
        assert result.metadata == []
        assert result.distances == []

    def test_search_results_from_chroma_empty(self):
        """Test SearchResults.from_chroma with empty ChromaDB results"""
        empty_chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        result = SearchResults.from_chroma(empty_chroma_results)
        
        assert result.is_empty()
        assert result.documents == []
        assert result.metadata == []
        assert result.distances == []
        assert result.error is None

    def test_search_with_course_name_resolution_failure(self, mock_vector_store):
        """Test search when course name resolution fails"""
        # Mock course resolution to return None
        mock_vector_store._resolve_course_name = Mock(return_value=None)
        
        result = mock_vector_store.search("test query", course_name="NonexistentCourse")
        
        assert result.error == "No course found matching 'NonexistentCourse'"
        assert result.is_empty()
        mock_vector_store.course_content.query.assert_not_called()

    def test_search_limit_precedence_over_max_results(self, mock_vector_store):
        """Test that explicit limit takes precedence over max_results"""
        mock_vector_store.course_content.query.return_value = {
            'documents': [['Test document']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        # Explicit limit should override max_results
        result = mock_vector_store.search("test query", limit=10)
        
        assert not result.error
        call_args = mock_vector_store.course_content.query.call_args[1]
        assert call_args['n_results'] == 10  # Not the default 5