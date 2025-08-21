import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store with test data"""
    mock_store = Mock(spec=VectorStore)

    # Mock search results for testing
    def mock_search(
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ):
        # Simulate different search scenarios
        if query == "test_empty":
            return SearchResults(documents=[], metadata=[], distances=[])
        elif query == "test_error":
            return SearchResults(
                documents=[], metadata=[], distances=[], error="Search failed"
            )
        elif query == "python basics":
            return SearchResults(
                documents=["Python is a programming language", "Variables in Python"],
                metadata=[
                    {"course_title": "Introduction to Python", "lesson_number": 1},
                    {"course_title": "Introduction to Python", "lesson_number": 2},
                ],
                distances=[0.1, 0.2],
            )
        elif course_name == "MCP" and query == "servers":
            return SearchResults(
                documents=["MCP servers handle protocol connections"],
                metadata=[
                    {"course_title": "Model Context Protocol", "lesson_number": 3}
                ],
                distances=[0.15],
            )
        else:
            # Default case - check if this is being overridden by the specific test
            return SearchResults(
                documents=["Default test content"],
                metadata=[{"course_title": "Test Course", "lesson_number": 1}],
                distances=[0.3],
            )

    mock_store.search.side_effect = mock_search

    # Mock course resolution
    def mock_resolve_course_name(course_name: str):
        mapping = {
            "MCP": "Model Context Protocol",
            "Python": "Introduction to Python",
            "nonexistent": None,
        }
        return mapping.get(course_name, "Test Course")

    mock_store._resolve_course_name.side_effect = mock_resolve_course_name

    # Mock lesson link retrieval
    def mock_get_lesson_link(course_title: str, lesson_number: int):
        if course_title == "Introduction to Python" and lesson_number == 1:
            return "https://example.com/python/lesson1"
        elif course_title == "Model Context Protocol" and lesson_number == 3:
            return "https://example.com/mcp/lesson3"
        return None

    mock_store.get_lesson_link.side_effect = mock_get_lesson_link

    # Mock course metadata
    mock_store.get_all_courses_metadata.return_value = [
        {
            "title": "Introduction to Python",
            "course_link": "https://example.com/python",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Python Basics"},
                {"lesson_number": 2, "lesson_title": "Variables and Data Types"},
            ],
        },
        {
            "title": "Model Context Protocol",
            "course_link": "https://example.com/mcp",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Introduction to MCP"},
                {"lesson_number": 2, "lesson_title": "MCP Architecture"},
                {"lesson_number": 3, "lesson_title": "MCP Servers"},
            ],
        },
    ]

    return mock_store


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create a CourseSearchTool with mocked vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store):
    """Create a CourseOutlineTool with mocked vector store"""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock_client = Mock()

    # Mock responses for different scenarios
    def mock_create(**kwargs):
        mock_response = Mock()

        # Check if tools are provided
        if "tools" in kwargs:
            # Simulate tool use response
            mock_response.stop_reason = "tool_use"
            mock_content_block = Mock()
            mock_content_block.type = "tool_use"
            mock_content_block.name = "search_course_content"
            mock_content_block.input = {"query": "test query"}
            mock_content_block.id = "tool_use_123"
            mock_response.content = [mock_content_block]
        else:
            # Simulate regular text response
            mock_response.stop_reason = "end_turn"
            mock_text_block = Mock()
            mock_text_block.text = (
                "This is the AI response based on the search results."
            )
            mock_response.content = [mock_text_block]

        return mock_response

    mock_client.messages.create.side_effect = mock_create
    return mock_client


@pytest.fixture
def mock_ai_generator(mock_anthropic_client):
    """Create an AIGenerator with mocked Anthropic client"""
    ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
    ai_gen.client = mock_anthropic_client
    return ai_gen


@pytest.fixture
def mock_config():
    """Create a mock configuration object"""
    config = Mock()
    config.CHUNK_SIZE = 1000
    config.CHUNK_OVERLAP = 200
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    config.MAX_HISTORY = 10
    return config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Introduction to Python",
        course_link="https://example.com/python",
        instructor="John Doe",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Python Basics",
                lesson_link="https://example.com/python/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Variables and Data Types",
                lesson_link="https://example.com/python/lesson2",
            ),
        ],
    )


@pytest.fixture
def sample_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a high-level programming language",
            course_title="Introduction to Python",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Variables store data values in Python",
            course_title="Introduction to Python",
            lesson_number=2,
            chunk_index=1,
        ),
    ]
