import anthropic
import time
import random
from typing import List, Optional, Dict, Any
from anthropic import APIError, RateLimitError
from anthropic._exceptions import OverloadedError

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **Course Content Search** - Use for questions about specific course content or detailed educational materials
2. **Course Outline** - Use for questions about course structure, lesson lists, or course overview information

Tool Selection Guidelines:
- **Course outline queries**: Use get_course_outline for questions about:
  - Course structure or lesson lists
  - "What lessons are in [course]?"
  - "Show me the outline for [course]"
  - Course overview or table of contents requests
- **Content-specific queries**: Use search_course_content for questions about:
  - Specific educational content within lessons
  - Technical details or explanations
  - Examples or code samples from lessons

Tool Usage Rules:
- **One tool call per query maximum**
- For outline queries, return the complete course title, course link, and numbered lesson list
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, max_retries: int = 3, retry_delay: float = 1.0, max_retry_delay: float = 60.0):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude with retry logic
        response = self._make_api_call_with_retry(api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use":
            if tool_manager:
                return self._handle_tool_execution(response, api_params, tool_manager)
            else:
                # No tool manager provided - return text indicating tools were requested but unavailable
                return "Tools were requested but no tool manager was provided."
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response with retry logic
        final_response = self._make_api_call_with_retry(final_params)
        return final_response.content[0].text
    
    def _make_api_call_with_retry(self, api_params: Dict[str, Any]):
        """
        Make API call with exponential backoff retry logic.
        
        Args:
            api_params: Parameters for the API call
            
        Returns:
            Response from the API
            
        Raises:
            APIError: After all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.messages.create(**api_params)
                
            except (OverloadedError, RateLimitError) as e:
                last_exception = e
                if attempt == self.max_retries:
                    # Final attempt failed
                    print(f"API call failed after {self.max_retries + 1} attempts: {e}")
                    raise
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.retry_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.max_retry_delay
                )
                
                print(f"API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                
            except APIError as e:
                # For other API errors, don't retry
                print(f"Non-retryable API error: {e}")
                raise
                
            except Exception as e:
                # For unexpected errors, don't retry
                print(f"Unexpected error in API call: {e}")
                raise
        
        # This shouldn't be reached, but just in case
        if last_exception:
            raise last_exception