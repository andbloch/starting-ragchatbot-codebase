import pytest
from unittest.mock import Mock
from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute method outputs"""

    def test_execute_successful_search(self, course_search_tool):
        """Test successful search with results"""
        result = course_search_tool.execute(query="python basics")
        
        # Verify the result is properly formatted
        assert "[Introduction to Python - Lesson 1]" in result
        assert "[Introduction to Python - Lesson 2]" in result
        assert "Python is a programming language" in result
        assert "Variables in Python" in result
        
        # Verify sources are tracked
        assert len(course_search_tool.last_sources) == 2
        assert course_search_tool.last_sources[0]["text"] == "Introduction to Python - Lesson 1"
        assert course_search_tool.last_sources[1]["text"] == "Introduction to Python - Lesson 2"

    def test_execute_with_course_filter(self, course_search_tool):
        """Test search with course name filter"""
        result = course_search_tool.execute(
            query="servers",
            course_name="MCP"
        )
        
        # Verify the result includes the course filter
        assert "[Model Context Protocol - Lesson 3]" in result
        assert "MCP servers handle protocol connections" in result
        
        # Verify sources include lesson link
        assert len(course_search_tool.last_sources) == 1
        source = course_search_tool.last_sources[0]
        assert source["text"] == "Model Context Protocol - Lesson 3"
        assert source["url"] == "https://example.com/mcp/lesson3"

    def test_execute_with_lesson_filter(self, course_search_tool):
        """Test search with lesson number filter"""
        result = course_search_tool.execute(
            query="python basics",
            lesson_number=1
        )
        
        # Should still return results (mock doesn't filter by lesson in this case)
        assert "[Introduction to Python - Lesson 1]" in result

    def test_execute_empty_results(self, course_search_tool):
        """Test handling of empty search results"""
        result = course_search_tool.execute(query="test_empty")
        
        assert result == "No relevant content found."
        assert course_search_tool.last_sources == []

    def test_execute_empty_results_with_filters(self, course_search_tool):
        """Test empty results with course and lesson filters"""
        result = course_search_tool.execute(
            query="test_empty",
            course_name="Python",
            lesson_number=5
        )
        
        assert "No relevant content found in course 'Python' in lesson 5." in result

    def test_execute_search_error(self, course_search_tool):
        """Test handling of search errors"""
        result = course_search_tool.execute(query="test_error")
        
        assert result == "Search failed"
        assert course_search_tool.last_sources == []

    def test_execute_format_results_no_lesson_number(self, mock_vector_store):
        """Test result formatting when lesson number is None"""
        # Create a mock result with None lesson number
        search_results = SearchResults(
            documents=["Test content without lesson"],
            metadata=[{"course_title": "Test Course", "lesson_number": None}],
            distances=[0.1]
        )
        # Override the side_effect for this specific test
        mock_vector_store.search.side_effect = None
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = None
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")
        
        assert "[Test Course]" in result  # No lesson number in header
        assert "Test content without lesson" in result
        
        # Verify source format
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"
        assert "url" not in tool.last_sources[0]

    def test_execute_source_tracking_with_links(self, course_search_tool):
        """Test that sources are properly tracked with lesson links"""
        result = course_search_tool.execute(query="python basics")
        
        # First source should have a link
        source1 = course_search_tool.last_sources[0]
        assert source1["text"] == "Introduction to Python - Lesson 1"
        assert source1["url"] == "https://example.com/python/lesson1"
        
        # Second source has no link (mock returns None)
        source2 = course_search_tool.last_sources[1]
        assert source2["text"] == "Introduction to Python - Lesson 2"
        assert "url" not in source2

    def test_execute_query_parameter_types(self, course_search_tool):
        """Test that execute method handles parameter types correctly"""
        # Test with all parameters
        result = course_search_tool.execute(
            query="test",
            course_name="Python",
            lesson_number=1
        )
        
        # Verify the mock vector store was called with correct parameters
        course_search_tool.store.search.assert_called_with(
            query="test",
            course_name="Python",
            lesson_number=1
        )

    def test_execute_optional_parameters(self, course_search_tool):
        """Test execute with optional parameters as None"""
        result = course_search_tool.execute(
            query="test",
            course_name=None,
            lesson_number=None
        )
        
        course_search_tool.store.search.assert_called_with(
            query="test",
            course_name=None,
            lesson_number=None
        )

    def test_format_results_consistency(self, course_search_tool):
        """Test that _format_results produces consistent output format"""
        # Test with multiple documents
        result = course_search_tool.execute(query="python basics")
        
        # Should have clear section separation
        sections = result.split("\n\n")
        assert len(sections) == 2  # Two documents = two sections
        
        # Each section should have header and content
        for section in sections:
            lines = section.split("\n")
            assert lines[0].startswith("[")  # Header
            assert lines[0].endswith("]")    # Header
            assert len(lines) >= 2           # Header + content

    def test_last_sources_reset_behavior(self, course_search_tool):
        """Test that last_sources is properly updated on each search"""
        # First search
        course_search_tool.execute(query="python basics")
        first_sources = course_search_tool.last_sources.copy()
        assert len(first_sources) == 2
        
        # Second search with different results
        course_search_tool.execute(query="servers", course_name="MCP")
        second_sources = course_search_tool.last_sources
        assert len(second_sources) == 1
        assert second_sources != first_sources
        
        # Empty search should clear sources
        course_search_tool.execute(query="test_empty")
        assert course_search_tool.last_sources == []

    def test_get_tool_definition(self, course_search_tool):
        """Test that get_tool_definition returns proper Anthropic tool format"""
        tool_def = course_search_tool.get_tool_definition()
        
        # Verify basic structure
        assert tool_def["name"] == "search_course_content"
        assert "description" in tool_def
        assert "input_schema" in tool_def
        
        # Verify schema structure
        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        
        # Verify required and optional parameters
        properties = schema["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties
        assert schema["required"] == ["query"]
        
        # Verify parameter types
        assert properties["query"]["type"] == "string"
        assert properties["course_name"]["type"] == "string"
        assert properties["lesson_number"]["type"] == "integer"

    def test_execute_with_invalid_search_limit(self, mock_vector_store):
        """Test handling of invalid search limits from vector store"""
        # Mock vector store to return error for invalid limit
        from vector_store import SearchResults
        # Override the side_effect to return our specific error
        mock_vector_store.search.side_effect = None
        mock_vector_store.search.return_value = SearchResults.empty("Invalid search limit: 0. Must be >= 1")
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")
        
        assert "Invalid search limit: 0. Must be >= 1" in result
        assert tool.last_sources == []