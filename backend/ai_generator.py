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
- **You may make multiple tool calls across up to 2 rounds** to gather comprehensive information
- **First round**: Use tools to gather initial information
- **Second round** (if needed): Use additional tools to gather more specific or related information
- For outline queries, return the complete course title, course link, and numbered lesson list
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool(s) first, then answer
- **Complex questions**: You may use multiple rounds of tool calls to gather comprehensive information
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
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with support for sequential tool calling.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool call rounds (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize conversation state
        messages = [{"role": "user", "content": query}]
        
        # Execute sequential tool calling rounds
        return self._execute_sequential_rounds(messages, system_content, tools, tool_manager, max_rounds)
    
    def _execute_sequential_rounds(self, messages: List[Dict], system_content: str, 
                                  tools: Optional[List], tool_manager, max_rounds: int) -> str:
        """
        Execute up to max_rounds of sequential tool calling.
        
        Termination conditions:
        1. max_rounds completed
        2. Claude's response has no tool_use blocks
        3. Tool execution fails
        
        Args:
            messages: List of conversation messages
            system_content: System prompt content
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of rounds to execute
            
        Returns:
            Final response as string
        """
        
        current_round = 0
        
        while current_round < max_rounds:
            current_round += 1
            
            # Prepare API parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            
            # Add tools only if we have them and a tool manager
            if tools and tool_manager:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Make API call
            try:
                response = self._make_api_call_with_retry(api_params)
            except Exception as e:
                # Tool execution failed - terminate
                return f"I encountered an error while processing your request: {str(e)}"
            
            # Check termination condition: no tool use
            if response.stop_reason != "tool_use":
                # Claude provided final response without tools
                return response.content[0].text
            
            # Handle tool execution for this round
            if not tool_manager:
                return "Tools were requested but no tool manager was provided."
            
            tool_results = self._execute_tools_for_round(response, tool_manager)
            
            # Check termination condition: tool execution failed
            if tool_results is None:
                return "I encountered an error while executing the requested tools."
            
            # Add AI's tool use response to conversation
            messages.append({"role": "assistant", "content": response.content})
            
            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})
            
            # Continue to next round if we haven't hit max_rounds
            # The while loop will handle the max_rounds termination
        
        # Max rounds reached - make final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
            # Deliberately no tools for final call
        }
        
        try:
            final_response = self._make_api_call_with_retry(final_params)
            return final_response.content[0].text
        except Exception as e:
            return f"I encountered an error while generating the final response: {str(e)}"
    
    def _execute_tools_for_round(self, response, tool_manager) -> Optional[List[Dict]]:
        """
        Execute all tool calls for a single round.
        
        Args:
            response: API response containing tool use requests
            tool_manager: Manager to execute tools
            
        Returns:
            List of tool results, or None if execution failed
        """
        
        tool_results = []
        
        try:
            for content_block in response.content:
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
            
            return tool_results if tool_results else None
            
        except Exception as e:
            print(f"Tool execution failed: {e}")
            return None
    
    
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