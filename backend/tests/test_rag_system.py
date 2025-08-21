from unittest.mock import MagicMock, Mock, patch

import pytest
from anthropic import APIError, RateLimitError
from anthropic._exceptions import OverloadedError
from rag_system import RAGSystem
from search_tools import ToolManager


class TestRAGSystem:
    """Test suite for RAG system handling of content-query related questions"""

    @pytest.fixture
    def mock_rag_system(self, mock_config):
        """Create a RAG system with mocked dependencies"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store_class,
            patch("rag_system.AIGenerator") as mock_ai_gen_class,
            patch("rag_system.SessionManager"),
            patch("rag_system.ToolManager") as mock_tool_manager_class,
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
        ):

            # Setup mock instances
            mock_vector_store = Mock()
            mock_vector_store_class.return_value = mock_vector_store

            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen

            mock_tool_manager = Mock()
            mock_tool_manager_class.return_value = mock_tool_manager

            # Create RAG system
            rag_system = RAGSystem(mock_config)

            # Override with our pre-configured mocks
            rag_system.vector_store = mock_vector_store
            rag_system.ai_generator = mock_ai_gen
            rag_system.tool_manager = mock_tool_manager

            return rag_system

    def test_query_basic_functionality(self, mock_rag_system):
        """Test basic query processing functionality"""
        # Setup mock responses
        mock_rag_system.ai_generator.generate_response.return_value = (
            "Python is a programming language"
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = [
            {
                "text": "Introduction to Python - Lesson 1",
                "url": "https://example.com/lesson1",
            }
        ]

        response, sources = mock_rag_system.query("What is Python?")

        # Verify the query was processed correctly
        assert response == "Python is a programming language"
        assert len(sources) == 1
        assert sources[0]["text"] == "Introduction to Python - Lesson 1"
        assert sources[0]["url"] == "https://example.com/lesson1"

        # Verify AI generator was called with correct parameters
        mock_rag_system.ai_generator.generate_response.assert_called_once()
        call_args = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert (
            call_args["query"]
            == "Answer this question about course materials: What is Python?"
        )
        assert call_args["tools"] == mock_rag_system.tool_manager.get_tool_definitions()
        assert call_args["tool_manager"] == mock_rag_system.tool_manager

    def test_query_with_session_id(self, mock_rag_system):
        """Test query processing with session context"""
        # Setup conversation history
        mock_rag_system.session_manager.get_conversation_history.return_value = (
            "Previous: Hello\nAI: Hi!"
        )
        mock_rag_system.ai_generator.generate_response.return_value = (
            "Response with context"
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        session_id = "test_session_123"
        response, sources = mock_rag_system.query(
            "Follow-up question", session_id=session_id
        )

        # Verify session history was retrieved and used
        mock_rag_system.session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )

        # Verify AI generator received history
        call_args = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert call_args["conversation_history"] == "Previous: Hello\nAI: Hi!"

        # Verify session was updated with new exchange
        mock_rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow-up question", "Response with context"
        )

    def test_query_content_specific_questions(self, mock_rag_system):
        """Test that content-specific questions trigger appropriate tool usage"""
        # Mock AI generator to simulate tool usage
        mock_rag_system.ai_generator.generate_response.return_value = (
            "Detailed explanation about variables"
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = [
            {"text": "Python Course - Lesson 2"}
        ]

        # Test content-specific query
        response, sources = mock_rag_system.query("How do variables work in Python?")

        # Verify tools were provided to AI generator
        call_args = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert call_args["tools"] is not None
        assert call_args["tool_manager"] is not None

        # Verify sources are returned
        assert len(sources) == 1
        assert sources[0]["text"] == "Python Course - Lesson 2"

    def test_query_course_outline_questions(self, mock_rag_system):
        """Test that course outline questions are handled properly"""
        # Mock response for course outline question
        mock_rag_system.ai_generator.generate_response.return_value = (
            "**Introduction to Python**\nLessons: 1. Basics, 2. Variables"
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = mock_rag_system.query(
            "What lessons are in the Python course?"
        )

        # Verify the query was processed as course outline request
        call_args = mock_rag_system.ai_generator.generate_response.call_args[1]
        assert "course materials" in call_args["query"]

        # Course outline typically doesn't return content sources
        assert len(sources) == 0

    def test_source_management(self, mock_rag_system):
        """Test that sources are properly managed and reset"""
        # Setup initial sources
        initial_sources = [
            {"text": "Course A - Lesson 1"},
            {"text": "Course B - Lesson 2"},
        ]
        mock_rag_system.tool_manager.get_last_sources.return_value = initial_sources
        mock_rag_system.ai_generator.generate_response.return_value = "Test response"

        response, sources = mock_rag_system.query("Test query")

        # Verify sources were retrieved and returned
        assert sources == initial_sources
        mock_rag_system.tool_manager.get_last_sources.assert_called_once()

        # Verify sources were reset after retrieval
        mock_rag_system.tool_manager.reset_sources.assert_called_once()

    def test_course_analytics(self, mock_rag_system):
        """Test course analytics functionality"""
        # Mock vector store responses
        mock_rag_system.vector_store.get_course_count.return_value = 3
        mock_rag_system.vector_store.get_existing_course_titles.return_value = [
            "Introduction to Python",
            "Model Context Protocol",
            "Web Development",
        ]

        analytics = mock_rag_system.get_course_analytics()

        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Introduction to Python" in analytics["course_titles"]

    def test_add_course_document(self, mock_rag_system):
        """Test adding a single course document"""
        from models import Course, Lesson

        # Mock processed course data
        test_course = Course(
            title="Test Course", lessons=[Lesson(lesson_number=1, title="Test Lesson")]
        )
        test_chunks = []  # Simplified for testing

        mock_rag_system.document_processor.process_course_document.return_value = (
            test_course,
            test_chunks,
        )

        course, chunk_count = mock_rag_system.add_course_document("/path/to/test.pdf")

        # Verify document processing was called
        mock_rag_system.document_processor.process_course_document.assert_called_once_with(
            "/path/to/test.pdf"
        )

        # Verify course was added to vector store
        mock_rag_system.vector_store.add_course_metadata.assert_called_once_with(
            test_course
        )
        mock_rag_system.vector_store.add_course_content.assert_called_once_with(
            test_chunks
        )

        assert course == test_course
        assert chunk_count == 0

    def test_add_course_document_error_handling(self, mock_rag_system):
        """Test error handling in add_course_document"""
        # Mock exception during processing
        mock_rag_system.document_processor.process_course_document.side_effect = (
            Exception("Processing failed")
        )

        course, chunk_count = mock_rag_system.add_course_document("/path/to/bad.pdf")

        # Should handle error gracefully
        assert course is None
        assert chunk_count == 0

    def test_query_prompt_formatting(self, mock_rag_system):
        """Test that query prompts are properly formatted"""
        mock_rag_system.ai_generator.generate_response.return_value = "Test response"
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        user_query = "Explain variables in Python"
        mock_rag_system.query(user_query)

        # Verify the prompt was formatted correctly
        call_args = mock_rag_system.ai_generator.generate_response.call_args[1]
        expected_prompt = f"Answer this question about course materials: {user_query}"
        assert call_args["query"] == expected_prompt

    def test_tool_registration(self, mock_rag_system):
        """Test that all necessary tools are registered"""
        # Verify search tool is registered
        assert hasattr(mock_rag_system, "search_tool")
        assert hasattr(mock_rag_system, "outline_tool")
        assert hasattr(mock_rag_system, "tool_manager")

        # Mock tool definitions to verify registration
        mock_rag_system.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content"},
            {"name": "get_course_outline"},
        ]

        tools = mock_rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tools]

        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_different_query_types(self, mock_rag_system):
        """Test RAG system response to different types of queries"""
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        # Test cases with different query types
        test_cases = [
            ("What is Python?", "general knowledge"),
            ("How do I define variables in the Python course?", "course content"),
            ("What lessons are in the MCP course?", "course structure"),
            ("Can you explain the concept from lesson 3?", "specific lesson"),
        ]

        for query, query_type in test_cases:
            mock_rag_system.ai_generator.generate_response.return_value = (
                f"Response to {query_type} question"
            )

            response, sources = mock_rag_system.query(query)

            # Verify each query type is processed
            assert response == f"Response to {query_type} question"

            # Verify AI generator was called with tools for all queries
            call_args = mock_rag_system.ai_generator.generate_response.call_args[1]
            assert call_args["tools"] is not None
            assert call_args["tool_manager"] is not None

    def test_session_isolation(self, mock_rag_system):
        """Test that different sessions maintain separate contexts"""
        mock_rag_system.ai_generator.generate_response.return_value = "Session response"
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        # Mock different conversation histories for different sessions
        def mock_get_history(session_id):
            histories = {
                "session1": "Session 1 history",
                "session2": "Session 2 history",
            }
            return histories.get(session_id)

        mock_rag_system.session_manager.get_conversation_history.side_effect = (
            mock_get_history
        )

        # Query with session 1
        mock_rag_system.query("Query 1", session_id="session1")
        call_args1 = mock_rag_system.ai_generator.generate_response.call_args[1]

        # Query with session 2
        mock_rag_system.query("Query 2", session_id="session2")
        call_args2 = mock_rag_system.ai_generator.generate_response.call_args[1]

        # Verify different histories were used
        assert "Session 1 history" in call_args1["conversation_history"]
        assert "Session 2 history" in call_args2["conversation_history"]

    def test_empty_sources_handling(self, mock_rag_system):
        """Test handling of queries that return no sources"""
        mock_rag_system.ai_generator.generate_response.return_value = (
            "No relevant content found"
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = mock_rag_system.query("Nonexistent topic")

        assert response == "No relevant content found"
        assert sources == []

    def test_multiple_source_types(self, mock_rag_system):
        """Test handling of different source types (with and without URLs)"""
        mixed_sources = [
            {"text": "Course A - Lesson 1", "url": "https://example.com/lesson1"},
            {"text": "Course B - Lesson 2"},  # No URL
            {"text": "Course C - Lesson 3", "url": "https://example.com/lesson3"},
        ]

        mock_rag_system.ai_generator.generate_response.return_value = (
            "Mixed sources response"
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = mixed_sources

        response, sources = mock_rag_system.query("Mixed sources query")

        assert len(sources) == 3
        assert "url" in sources[0]  # First has URL
        assert "url" not in sources[1]  # Second has no URL
        assert "url" in sources[2]  # Third has URL

    def test_api_overloaded_error_handling(self, mock_rag_system):
        """Test graceful handling of API overloaded errors"""
        # Mock AI generator to raise OverloadedError
        mock_response = Mock()
        mock_request = Mock()
        mock_rag_system.ai_generator.generate_response.side_effect = OverloadedError(
            "API overloaded", response=mock_response, body={}
        )

        response, sources = mock_rag_system.query("Test query causing overload")

        # Should return user-friendly error message, not crash
        assert "experiencing high demand" in response
        assert "temporarily overloaded" in response
        assert sources == []

        # Should still call the AI generator (which fails)
        mock_rag_system.ai_generator.generate_response.assert_called_once()

    def test_api_rate_limit_error_handling(self, mock_rag_system):
        """Test graceful handling of API rate limit errors"""
        # Mock AI generator to raise RateLimitError
        mock_response = Mock()
        mock_request = Mock()
        mock_rag_system.ai_generator.generate_response.side_effect = RateLimitError(
            "Rate limited", response=mock_response, body={}
        )

        response, sources = mock_rag_system.query("Test query causing rate limit")

        # Should return user-friendly error message
        assert "experiencing high demand" in response
        assert "temporarily overloaded" in response
        assert sources == []

    def test_api_authentication_error_handling(self, mock_rag_system):
        """Test graceful handling of API authentication errors"""
        # Mock AI generator to raise APIError
        mock_request = Mock()
        mock_rag_system.ai_generator.generate_response.side_effect = APIError(
            "Authentication failed", request=mock_request, body={}
        )

        response, sources = mock_rag_system.query("Test query with auth error")

        # Should return configuration error message
        assert "trouble connecting to the AI service" in response
        assert "check your configuration" in response
        assert sources == []

    def test_unexpected_error_handling(self, mock_rag_system):
        """Test graceful handling of unexpected errors"""
        # Mock AI generator to raise unexpected error
        mock_rag_system.ai_generator.generate_response.side_effect = ValueError(
            "Unexpected error"
        )

        response, sources = mock_rag_system.query("Test query causing unexpected error")

        # Should return generic error message
        assert "unexpected error" in response
        assert "try again" in response
        assert sources == []

    def test_error_handling_preserves_session_isolation(self, mock_rag_system):
        """Test that error handling doesn't break session isolation"""
        # Mock AI generator to fail
        mock_response = Mock()
        mock_rag_system.ai_generator.generate_response.side_effect = OverloadedError(
            "API overloaded", response=mock_response, body={}
        )

        # Query with session ID should not update history on error
        response, sources = mock_rag_system.query(
            "Error query", session_id="test_session"
        )

        # Should return error message
        assert "experiencing high demand" in response
        assert sources == []

        # Session should NOT be updated on error
        mock_rag_system.session_manager.add_exchange.assert_not_called()

    def test_error_recovery_after_failure(self, mock_rag_system):
        """Test that system recovers after API errors"""
        # First query fails
        mock_response = Mock()
        mock_rag_system.ai_generator.generate_response.side_effect = OverloadedError(
            "API overloaded", response=mock_response, body={}
        )

        response1, sources1 = mock_rag_system.query("First query that fails")
        assert "experiencing high demand" in response1
        assert sources1 == []

        # Second query succeeds
        mock_rag_system.ai_generator.generate_response.side_effect = None
        mock_rag_system.ai_generator.generate_response.return_value = (
            "Success after recovery"
        )
        mock_rag_system.tool_manager.get_last_sources.return_value = [
            {"text": "Test source"}
        ]

        response2, sources2 = mock_rag_system.query("Second query after recovery")
        assert response2 == "Success after recovery"
        assert len(sources2) == 1

    def test_api_error_logging(self, mock_rag_system, capsys):
        """Test that API errors are properly logged"""
        # Mock AI generator to raise OverloadedError
        mock_response = Mock()
        mock_rag_system.ai_generator.generate_response.side_effect = OverloadedError(
            "API overloaded", response=mock_response, body={}
        )

        mock_rag_system.query("Test query for logging")

        # Verify error was logged
        captured = capsys.readouterr()
        assert "API overload/rate limit error handled gracefully" in captured.out
