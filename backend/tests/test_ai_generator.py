import pytest
from unittest.mock import Mock, patch, call
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from anthropic import RateLimitError, APIError
from anthropic._exceptions import OverloadedError


class TestAIGenerator:
    """Test suite for AIGenerator tool calling functionality"""

    def test_generate_response_without_tools(self, mock_ai_generator):
        """Test basic response generation without tools"""
        response = mock_ai_generator.generate_response("What is Python?")
        
        # Verify the API was called with correct parameters
        mock_ai_generator.client.messages.create.assert_called_once()
        call_args = mock_ai_generator.client.messages.create.call_args[1]
        
        assert call_args["model"] == "claude-3-sonnet-20240229"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"] == [{"role": "user", "content": "What is Python?"}]
        assert call_args["system"] == mock_ai_generator.SYSTEM_PROMPT
        assert "tools" not in call_args

    def test_generate_response_with_tools(self, mock_ai_generator, course_search_tool):
        """Test response generation with tools provided"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        
        # Mock the tool manager's execute_tool method
        tool_manager.execute_tool = Mock(return_value="Search results here")
        
        # Configure mock to use tool once, then provide final response
        call_count = 0
        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = Mock()
            
            if call_count == 1:
                # First round - use tool
                mock_response.stop_reason = "tool_use"
                mock_content_block = Mock()
                mock_content_block.type = "tool_use"
                mock_content_block.name = "search_course_content"
                mock_content_block.input = {"query": "test"}
                mock_content_block.id = "tool_use_1"
                mock_response.content = [mock_content_block]
            else:
                # Second round - final response without tools
                mock_response.stop_reason = "end_turn"
                mock_text_block = Mock()
                mock_text_block.text = "Here's the answer based on the search."
                mock_response.content = [mock_text_block]
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = mock_create
        
        response = mock_ai_generator.generate_response(
            query="What is Python?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify API calls were made (tool use + final response)
        assert mock_ai_generator.client.messages.create.call_count == 2
        
        # Verify the first API call (initial) had tools
        first_call_args = mock_ai_generator.client.messages.create.call_args_list[0][1]
        assert "tools" in first_call_args
        assert first_call_args["tool_choice"] == {"type": "auto"}
        
        # Verify tool was executed once
        tool_manager.execute_tool.assert_called_once()
        
        # Verify final response
        assert response == "Here's the answer based on the search."

    def test_handle_tool_execution_flow(self, mock_ai_generator, course_search_tool):
        """Test the complete tool execution flow"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        
        # Mock tool execution to return specific results
        tool_manager.execute_tool = Mock(return_value="Python is a programming language")
        
        # Configure mock to simulate tool use response first, then final response
        def mock_create(**kwargs):
            mock_response = Mock()
            if "tools" in kwargs and len(kwargs.get("messages", [])) == 1:
                # First call with tools - return tool use
                mock_response.stop_reason = "tool_use"
                mock_content_block = Mock()
                mock_content_block.type = "tool_use"
                mock_content_block.name = "search_course_content"
                mock_content_block.input = {"query": "python basics"}
                mock_content_block.id = "tool_use_123"
                mock_response.content = [mock_content_block]
            else:
                # Final call without tools - return text response
                mock_response.stop_reason = "end_turn"
                mock_text_block = Mock()
                mock_text_block.text = "Based on the search, Python is a programming language."
                mock_response.content = [mock_text_block]
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = mock_create
        
        response = mock_ai_generator.generate_response(
            query="Tell me about Python",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="python basics"
        )
        
        # Verify final response
        assert response == "Based on the search, Python is a programming language."
        
        # Verify the API was called twice (initial + follow-up)
        assert mock_ai_generator.client.messages.create.call_count == 2

    def test_handle_tool_execution_with_conversation_history(self, mock_ai_generator, course_search_tool):
        """Test tool execution with conversation history context"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        tool_manager.execute_tool = Mock(return_value="Search results")
        
        conversation_history = "User: Hello\nAI: Hi there!"
        
        response = mock_ai_generator.generate_response(
            query="What is Python?",
            conversation_history=conversation_history,
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify conversation history was included in system prompt
        call_args = mock_ai_generator.client.messages.create.call_args[1]
        expected_system = f"{mock_ai_generator.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
        assert call_args["system"] == expected_system

    def test_tool_execution_multiple_tools(self, mock_ai_generator, course_search_tool, course_outline_tool):
        """Test handling of multiple tool calls in one response"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        tool_manager.register_tool(course_outline_tool)
        
        # Mock tool executions
        def mock_execute_tool(tool_name, **kwargs):
            if tool_name == "search_course_content":
                return "Search results for content"
            elif tool_name == "get_course_outline":
                return "Course outline results"
            return "Unknown tool"
        
        tool_manager.execute_tool = Mock(side_effect=mock_execute_tool)
        
        # Configure mock to return multiple tool uses
        def mock_create(**kwargs):
            mock_response = Mock()
            if "tools" in kwargs and len(kwargs.get("messages", [])) == 1:
                # First call - return multiple tool uses
                mock_response.stop_reason = "tool_use"
                
                # First tool use
                mock_content_block1 = Mock()
                mock_content_block1.type = "tool_use"
                mock_content_block1.name = "search_course_content"
                mock_content_block1.input = {"query": "python"}
                mock_content_block1.id = "tool_use_1"
                
                # Second tool use
                mock_content_block2 = Mock()
                mock_content_block2.type = "tool_use"
                mock_content_block2.name = "get_course_outline"
                mock_content_block2.input = {"course_title": "Python"}
                mock_content_block2.id = "tool_use_2"
                
                mock_response.content = [mock_content_block1, mock_content_block2]
            else:
                # Final response
                mock_response.stop_reason = "end_turn"
                mock_text_block = Mock()
                mock_text_block.text = "Here's the information you requested."
                mock_response.content = [mock_text_block]
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = mock_create
        
        response = mock_ai_generator.generate_response(
            query="Tell me about Python course",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_has_calls([
            call("search_course_content", query="python"),
            call("get_course_outline", course_title="Python")
        ])

    def test_system_prompt_content(self):
        """Test that the system prompt contains expected instructions"""
        system_prompt = AIGenerator.SYSTEM_PROMPT
        
        # Verify key instruction elements
        assert "Course Content Search" in system_prompt
        assert "Course Outline" in system_prompt
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "You may make multiple tool calls across up to 2 rounds" in system_prompt
        assert "No meta-commentary" in system_prompt

    def test_base_parameters_configuration(self):
        """Test that base parameters are properly configured"""
        ai_gen = AIGenerator("test_key", "claude-3-sonnet-20240229")
        
        expected_params = {
            "model": "claude-3-sonnet-20240229",
            "temperature": 0,
            "max_tokens": 800
        }
        
        assert ai_gen.base_params == expected_params

    def test_tool_result_message_structure(self, mock_ai_generator, course_search_tool):
        """Test that tool results are properly structured in follow-up messages"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        tool_manager.execute_tool = Mock(return_value="Tool execution result")
        
        # Track the actual API calls to verify message structure
        api_calls = []
        
        def capture_create(**kwargs):
            api_calls.append(kwargs)
            mock_response = Mock()
            
            if len(api_calls) == 1:
                # First call - return tool use
                mock_response.stop_reason = "tool_use"
                mock_content_block = Mock()
                mock_content_block.type = "tool_use"
                mock_content_block.name = "search_course_content"
                mock_content_block.input = {"query": "test"}
                mock_content_block.id = "tool_123"
                mock_response.content = [mock_content_block]
            else:
                # Second call - final response
                mock_response.stop_reason = "end_turn"
                mock_text_block = Mock()
                mock_text_block.text = "Final response"
                mock_response.content = [mock_text_block]
            
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = capture_create
        
        response = mock_ai_generator.generate_response(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify two API calls were made
        assert len(api_calls) == 2
        
        # Verify the second call has the proper message structure
        second_call = api_calls[1]
        messages = second_call["messages"]
        
        # Should have: user message, assistant tool use, user tool result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Verify tool result structure
        tool_result_content = messages[2]["content"]
        assert len(tool_result_content) == 1
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "tool_123"
        assert tool_result_content[0]["content"] == "Tool execution result"

    def test_no_tool_manager_provided(self, mock_ai_generator, course_search_tool):
        """Test behavior when tool_manager is not provided but tools are present"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        
        # Configure to return tool_use but no tool_manager provided
        def mock_create(**kwargs):
            mock_response = Mock()
            mock_response.stop_reason = "tool_use"
            mock_content_block = Mock()
            mock_content_block.type = "tool_use"
            mock_response.content = [mock_content_block]
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = mock_create
        
        # Should not crash and should handle gracefully
        response = mock_ai_generator.generate_response(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=None  # No tool manager provided
        )
        
        # Should return error message since tool_manager is None but tools were requested
        assert response == "Tools were requested but no tool manager was provided."

    def test_conversation_history_formatting(self, mock_ai_generator):
        """Test that conversation history is properly formatted in system content"""
        # Test with conversation history
        history = "Previous conversation content"
        
        mock_ai_generator.generate_response(
            query="Test query",
            conversation_history=history
        )
        
        call_args = mock_ai_generator.client.messages.create.call_args[1]
        expected_system = f"{mock_ai_generator.SYSTEM_PROMPT}\n\nPrevious conversation:\n{history}"
        assert call_args["system"] == expected_system
        
        # Reset mock
        mock_ai_generator.client.messages.create.reset_mock()
        
        # Test without conversation history
        mock_ai_generator.generate_response(query="Test query")
        
        call_args = mock_ai_generator.client.messages.create.call_args[1]
        assert call_args["system"] == mock_ai_generator.SYSTEM_PROMPT

    def test_api_retry_on_overloaded_error(self):
        """Test retry logic for OverloadedError"""
        ai_gen = AIGenerator("test_key", "claude-3-sonnet-20240229", max_retries=2, retry_delay=0.1)
        
        # Mock client to fail twice then succeed
        success_response = Mock()
        success_response.stop_reason = "end_turn"
        success_response.content = [Mock(text="Success after retry")]
        
        call_count = 0
        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                error_response = Mock()
                raise OverloadedError("API overloaded", response=error_response, body={})
            return success_response
        
        ai_gen.client.messages.create = Mock(side_effect=mock_create)
        
        # Should succeed after retries
        result = ai_gen.generate_response("Test query")
        
        assert result == "Success after retry"
        assert ai_gen.client.messages.create.call_count == 3  # 2 failures + 1 success

    def test_api_retry_on_rate_limit_error(self):
        """Test retry logic for RateLimitError"""
        ai_gen = AIGenerator("test_key", "claude-3-sonnet-20240229", max_retries=1, retry_delay=0.1)
        
        # Mock client to fail once then succeed
        success_response = Mock()
        success_response.stop_reason = "end_turn"
        success_response.content = [Mock(text="Success after rate limit")]
        
        call_count = 0
        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error_response = Mock()
                raise RateLimitError("Rate limited", response=error_response, body={})
            return success_response
        
        ai_gen.client.messages.create = Mock(side_effect=mock_create)
        
        result = ai_gen.generate_response("Test query")
        
        assert result == "Success after rate limit"
        assert ai_gen.client.messages.create.call_count == 2

    def test_api_retry_exhausted(self):
        """Test behavior when all retries are exhausted"""
        ai_gen = AIGenerator("test_key", "claude-3-sonnet-20240229", max_retries=1, retry_delay=0.1)
        
        # Mock client to always fail
        def mock_create(**kwargs):
            mock_response = Mock()
            raise OverloadedError("Persistent overload", response=mock_response, body={})
        
        ai_gen.client.messages.create = Mock(side_effect=mock_create)
        
        # Should return error message after exhausting retries
        result = ai_gen.generate_response("Test query")
        assert "I encountered an error while processing your request" in result
        
        assert ai_gen.client.messages.create.call_count == 2  # max_retries + 1

    def test_api_no_retry_on_non_retryable_error(self):
        """Test that non-retryable errors don't trigger retries"""
        ai_gen = AIGenerator("test_key", "claude-3-sonnet-20240229", max_retries=3, retry_delay=0.1)
        
        # Mock client to raise non-retryable error
        def mock_create(**kwargs):
            mock_request = Mock()
            raise APIError("Authentication failed", request=mock_request, body={})
        
        ai_gen.client.messages.create = Mock(side_effect=mock_create)
        
        # Should return error message immediately without retries
        result = ai_gen.generate_response("Test query")
        assert "I encountered an error while processing your request" in result
        
        assert ai_gen.client.messages.create.call_count == 1  # No retries

    def test_retry_configuration(self):
        """Test that retry configuration is properly set"""
        ai_gen = AIGenerator(
            "test_key", 
            "claude-3-sonnet-20240229", 
            max_retries=5, 
            retry_delay=2.0, 
            max_retry_delay=120.0
        )
        
        assert ai_gen.max_retries == 5
        assert ai_gen.retry_delay == 2.0
        assert ai_gen.max_retry_delay == 120.0

    def test_retry_in_tool_execution_flow(self):
        """Test retry logic works in tool execution flow"""
        ai_gen = AIGenerator("test_key", "claude-3-sonnet-20240229", max_retries=1, retry_delay=0.1)
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool result"
        
        # First call succeeds with tool use, second call (follow-up) fails then succeeds
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [Mock(type="tool_use", name="test_tool", input={}, id="123")]
        
        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [Mock(text="Final response")]
        
        call_count = 0
        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            elif call_count == 2:
                mock_response = Mock()
                raise OverloadedError("Overloaded on follow-up", response=mock_response, body={})
            else:
                return second_response
        
        ai_gen.client.messages.create = Mock(side_effect=mock_create)
        
        result = ai_gen.generate_response("Test query", tools=[{}], tool_manager=tool_manager)
        
        assert result == "Final response"
        assert ai_gen.client.messages.create.call_count == 3  # First + retry on second

    def test_sequential_tool_calling_two_rounds(self, mock_ai_generator, course_search_tool, course_outline_tool):
        """Test sequential tool calling across 2 rounds"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        tool_manager.register_tool(course_outline_tool)
        
        # Mock tool executions
        def mock_execute_tool(tool_name, **kwargs):
            if tool_name == "get_course_outline":
                return "Course X has Lesson 4: Python Basics"
            elif tool_name == "search_course_content":
                return "Found Course Y that covers Python Basics"
            return "Unknown tool result"
        
        tool_manager.execute_tool = Mock(side_effect=mock_execute_tool)
        
        # Configure mock to simulate 2 rounds of tool calls then final response
        call_count = 0
        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = Mock()
            
            if call_count == 1:
                # First round - get course outline
                mock_response.stop_reason = "tool_use"
                mock_content_block = Mock()
                mock_content_block.type = "tool_use"
                mock_content_block.name = "get_course_outline"
                mock_content_block.input = {"course_title": "Course X"}
                mock_content_block.id = "tool_use_1"
                mock_response.content = [mock_content_block]
            elif call_count == 2:
                # Second round - search for similar course
                mock_response.stop_reason = "tool_use"
                mock_content_block = Mock()
                mock_content_block.type = "tool_use"
                mock_content_block.name = "search_course_content"
                mock_content_block.input = {"query": "Python Basics"}
                mock_content_block.id = "tool_use_2"
                mock_response.content = [mock_content_block]
            else:
                # Final response after 2 rounds
                mock_response.stop_reason = "end_turn"
                mock_text_block = Mock()
                mock_text_block.text = "Course Y covers the same topic as Lesson 4 of Course X."
                mock_response.content = [mock_text_block]
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = mock_create
        
        response = mock_ai_generator.generate_response(
            query="Find a course that discusses the same topic as lesson 4 of Course X",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_has_calls([
            call("get_course_outline", course_title="Course X"),
            call("search_course_content", query="Python Basics")
        ])
        
        # Verify final response
        assert response == "Course Y covers the same topic as Lesson 4 of Course X."
        
        # Verify 3 API calls were made (2 tool rounds + 1 final)
        assert mock_ai_generator.client.messages.create.call_count == 3

    def test_sequential_tool_calling_terminates_on_non_tool_response(self, mock_ai_generator, course_search_tool):
        """Test that sequential calling terminates when Claude doesn't use tools"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        tool_manager.execute_tool = Mock(return_value="Search results")
        
        # Configure mock to use tool once, then provide final response
        call_count = 0
        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = Mock()
            
            if call_count == 1:
                # First round - use tool
                mock_response.stop_reason = "tool_use"
                mock_content_block = Mock()
                mock_content_block.type = "tool_use"
                mock_content_block.name = "search_course_content"
                mock_content_block.input = {"query": "test"}
                mock_content_block.id = "tool_use_1"
                mock_response.content = [mock_content_block]
            else:
                # Second round - final response without tools
                mock_response.stop_reason = "end_turn"
                mock_text_block = Mock()
                mock_text_block.text = "Here's the answer based on the search."
                mock_response.content = [mock_text_block]
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = mock_create
        
        response = mock_ai_generator.generate_response(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify only one tool execution
        assert tool_manager.execute_tool.call_count == 1
        
        # Verify final response
        assert response == "Here's the answer based on the search."
        
        # Verify only 2 API calls (tool use + final response)
        assert mock_ai_generator.client.messages.create.call_count == 2

    def test_sequential_tool_calling_max_rounds_reached(self, mock_ai_generator, course_search_tool):
        """Test behavior when max rounds is reached"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        tool_manager.execute_tool = Mock(return_value="Search results")
        
        # Configure mock to always return tool use responses
        def mock_create(**kwargs):
            mock_response = Mock()
            if "tools" in kwargs:
                # Tool use response
                mock_response.stop_reason = "tool_use"
                mock_content_block = Mock()
                mock_content_block.type = "tool_use"
                mock_content_block.name = "search_course_content"
                mock_content_block.input = {"query": "test"}
                mock_content_block.id = "tool_use_1"
                mock_response.content = [mock_content_block]
            else:
                # Final response without tools
                mock_response.stop_reason = "end_turn"
                mock_text_block = Mock()
                mock_text_block.text = "Final response after max rounds."
                mock_response.content = [mock_text_block]
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = mock_create
        
        response = mock_ai_generator.generate_response(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
            max_rounds=2
        )
        
        # Verify tool was executed twice (max rounds)
        assert tool_manager.execute_tool.call_count == 2
        
        # Verify final response
        assert response == "Final response after max rounds."
        
        # Verify 3 API calls (2 tool rounds + 1 final without tools)
        assert mock_ai_generator.client.messages.create.call_count == 3

    def test_sequential_tool_calling_tool_execution_failure(self, mock_ai_generator, course_search_tool):
        """Test behavior when tool execution fails in sequential calling"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        
        # Mock tool manager to raise exception
        tool_manager.execute_tool = Mock(side_effect=Exception("Tool execution failed"))
        
        # Configure mock to return tool use
        def mock_create(**kwargs):
            mock_response = Mock()
            mock_response.stop_reason = "tool_use"
            mock_content_block = Mock()
            mock_content_block.type = "tool_use"
            mock_content_block.name = "search_course_content"
            mock_content_block.input = {"query": "test"}
            mock_content_block.id = "tool_use_1"
            mock_response.content = [mock_content_block]
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = mock_create
        
        response = mock_ai_generator.generate_response(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify tool execution was attempted
        assert tool_manager.execute_tool.call_count == 1
        
        # Verify error response
        assert response == "I encountered an error while executing the requested tools."
        
        # Verify only 1 API call (failed on first tool round)
        assert mock_ai_generator.client.messages.create.call_count == 1

    def test_sequential_tool_calling_conversation_context_preserved(self, mock_ai_generator, course_search_tool):
        """Test that conversation context is preserved across sequential rounds"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        tool_manager.execute_tool = Mock(return_value="Search results")
        
        conversation_history = "User: Previous question\nAI: Previous answer"
        
        # Track the actual API calls to verify message structure
        api_calls = []
        
        def capture_create(**kwargs):
            api_calls.append(kwargs)
            mock_response = Mock()
            
            if len(api_calls) == 1:
                # First call - return tool use
                mock_response.stop_reason = "tool_use"
                mock_content_block = Mock()
                mock_content_block.type = "tool_use"
                mock_content_block.name = "search_course_content"
                mock_content_block.input = {"query": "test"}
                mock_content_block.id = "tool_123"
                mock_response.content = [mock_content_block]
            else:
                # Subsequent calls - final response
                mock_response.stop_reason = "end_turn"
                mock_text_block = Mock()
                mock_text_block.text = "Final response"
                mock_response.content = [mock_text_block]
            
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = capture_create
        
        response = mock_ai_generator.generate_response(
            query="Test query",
            conversation_history=conversation_history,
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify conversation history was included in both calls
        for call_args in api_calls:
            expected_system = f"{mock_ai_generator.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            assert call_args["system"] == expected_system
        
        # Verify we had exactly 2 API calls (tool use + final response)  
        assert len(api_calls) == 2
        
        # The sequential implementation accumulates messages across rounds
        # Both calls should preserve conversation context
        assert len(api_calls[0]["messages"]) >= 1  # At least the initial message
        assert len(api_calls[1]["messages"]) >= 3  # Initial + assistant tool use + tool result

    def test_max_rounds_parameter_customization(self, mock_ai_generator, course_search_tool):
        """Test that max_rounds parameter can be customized"""
        tool_manager = ToolManager()
        tool_manager.register_tool(course_search_tool)
        tool_manager.execute_tool = Mock(return_value="Search results")
        
        # Configure mock to always return tool use responses
        def mock_create(**kwargs):
            mock_response = Mock()
            if "tools" in kwargs:
                mock_response.stop_reason = "tool_use"
                mock_content_block = Mock()
                mock_content_block.type = "tool_use"
                mock_content_block.name = "search_course_content"
                mock_content_block.input = {"query": "test"}
                mock_content_block.id = "tool_use_1"
                mock_response.content = [mock_content_block]
            else:
                mock_response.stop_reason = "end_turn"
                mock_text_block = Mock()
                mock_text_block.text = "Final response"
                mock_response.content = [mock_text_block]
            return mock_response
        
        mock_ai_generator.client.messages.create.side_effect = mock_create
        
        # Test with max_rounds=1
        response = mock_ai_generator.generate_response(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
            max_rounds=1
        )
        
        # Verify only one tool execution
        assert tool_manager.execute_tool.call_count == 1
        
        # Verify 2 API calls (1 tool round + 1 final)
        assert mock_ai_generator.client.messages.create.call_count == 2