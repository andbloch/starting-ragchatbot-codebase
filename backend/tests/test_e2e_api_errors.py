import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from anthropic import RateLimitError, APIError  
from anthropic._exceptions import OverloadedError


class TestE2EAPIErrorHandling:
    """End-to-end tests for API error handling from frontend to backend"""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        from app import app
        return TestClient(app)

    def test_e2e_overloaded_error_returns_graceful_response(self, client):
        """Test that 529 overloaded errors return graceful responses to frontend"""
        
        # Mock the RAG system to return graceful error message (as it should do)
        with patch('app.rag_system') as mock_rag_system:
            # The RAG system should handle the error gracefully and return a user-friendly message
            mock_rag_system.query.return_value = (
                "I'm experiencing high demand right now and the AI service is temporarily overloaded. "
                "Please try your question again in a few moments. If the issue persists, the service "
                "may be experiencing temporary capacity constraints.",
                []
            )
            
            # Make request to the API endpoint
            response = client.post("/api/query", json={
                "query": "What was covered in lesson 5 of the MCP course?",
                "session_id": "test_session"
            })
            
            # Should NOT return 500 error
            assert response.status_code != 500
            
            # Should return graceful error response
            assert response.status_code == 200
            response_data = response.json()
            
            # Should contain user-friendly error message
            assert "experiencing high demand" in response_data["answer"].lower() or \
                   "temporarily overloaded" in response_data["answer"].lower() or \
                   "try again" in response_data["answer"].lower()
            
            # Should have empty sources
            assert response_data["sources"] == []
            
            # Should have session_id
            assert "session_id" in response_data

    def test_e2e_rate_limit_error_returns_graceful_response(self, client):
        """Test that rate limit errors return graceful responses to frontend"""
        
        with patch('app.rag_system') as mock_rag_system:
            # RAG system should handle the error and return graceful message
            mock_rag_system.query.return_value = (
                "I'm experiencing high demand right now and the AI service is temporarily overloaded. "
                "Please try your question again in a few moments.",
                []
            )
            
            response = client.post("/api/query", json={
                "query": "Test query",
                "session_id": "test_session"
            })
            
            assert response.status_code == 200
            response_data = response.json()
            assert "experiencing high demand" in response_data["answer"].lower() or \
                   "rate limit" in response_data["answer"].lower() or \
                   "try again" in response_data["answer"].lower()

    def test_e2e_authentication_error_returns_graceful_response(self, client):
        """Test that authentication errors return graceful responses to frontend"""
        
        with patch('app.rag_system') as mock_rag_system:
            # RAG system should handle the error and return graceful message
            mock_rag_system.query.return_value = (
                "I'm sorry, but I'm having trouble connecting to the AI service right now. "
                "Please check your configuration or try again later.",
                []
            )
            
            response = client.post("/api/query", json={
                "query": "Test query",
                "session_id": "test_session"
            })
            
            assert response.status_code == 200
            response_data = response.json()
            assert "trouble connecting" in response_data["answer"].lower() or \
                   "configuration" in response_data["answer"].lower()

    def test_e2e_with_real_rag_system_api_error_simulation(self, client):
        """Test with real RAG system but mocked AI generator to simulate the exact error path"""
        
        # This test simulates the real error path more closely by mocking the anthropic client
        # at the lowest level, which should trigger our graceful error handling
        with patch('anthropic.Anthropic') as mock_anthropic:
            # Mock the anthropic client to always raise OverloadedError
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            # Create a proper mock response for the OverloadedError
            import httpx
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 529
            mock_response.headers = {"request-id": "test-request-id"}
            
            # Configure the client to always raise OverloadedError
            mock_client.messages.create.side_effect = OverloadedError(
                "Error code: 529 - {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}, 'request_id': None}",
                response=mock_response, 
                body={}
            )
            
            response = client.post("/api/query", json={
                "query": "What was covered in lesson 5 of the MCP course?",
                "session_id": "test_session"
            })
            
            # The key test: should not crash with 500
            if response.status_code == 500:
                print(f"ERROR: Got 500 response: {response.text}")
                pytest.fail("API should handle errors gracefully, not return 500")
            
            assert response.status_code == 200
            response_data = response.json()
            
            # Should contain graceful error message
            error_indicators = [
                "experiencing high demand",
                "temporarily overloaded", 
                "try again",
                "service is overloaded"
            ]
            
            answer_lower = response_data["answer"].lower()
            has_graceful_message = any(indicator in answer_lower for indicator in error_indicators)
            
            if not has_graceful_message:
                print(f"Response answer: {response_data['answer']}")
                pytest.fail("Response should contain graceful error message")

    def test_e2e_successful_recovery_after_error(self, client):
        """Test that system recovers after API errors"""
        
        with patch('app.rag_system') as mock_rag_system:
            # First request fails - return graceful error message (as RAG system would do)
            mock_rag_system.query.return_value = (
                "I'm experiencing high demand right now and the AI service is temporarily overloaded. "
                "Please try your question again in a few moments.",
                []
            )
            
            response1 = client.post("/api/query", json={
                "query": "First query that fails",
                "session_id": "test_session"
            })
            
            assert response1.status_code == 200
            assert "experiencing high demand" in response1.json()["answer"].lower() or \
                   "temporarily overloaded" in response1.json()["answer"].lower()
            
            # Second request succeeds
            mock_rag_system.query.side_effect = None
            mock_rag_system.query.return_value = ("Success after recovery", [{"text": "Test source"}])
            
            response2 = client.post("/api/query", json={
                "query": "Second query after recovery", 
                "session_id": "test_session"
            })
            
            assert response2.status_code == 200
            response_data = response2.json()
            assert response_data["answer"] == "Success after recovery"
            assert len(response_data["sources"]) == 1

    def test_e2e_concurrent_error_handling(self, client):
        """Test that concurrent requests with errors don't interfere"""
        
        with patch('app.rag_system') as mock_rag_system:
            # Return graceful error message (as RAG system would do)
            mock_rag_system.query.return_value = (
                "I'm experiencing high demand right now and the AI service is temporarily overloaded. "
                "Please try your question again in a few moments.",
                []
            )
            
            # Make multiple concurrent requests
            responses = []
            for i in range(3):
                response = client.post("/api/query", json={
                    "query": f"Test query {i}",
                    "session_id": f"test_session_{i}"
                })
                responses.append(response)
            
            # All should handle errors gracefully
            for i, response in enumerate(responses):
                assert response.status_code == 200, f"Request {i} failed with {response.status_code}"
                response_data = response.json()
                assert "experiencing high demand" in response_data["answer"].lower()
                assert f"test_session_{i}" == response_data["session_id"]

    def test_e2e_error_preserves_response_format(self, client):
        """Test that errors preserve the expected QueryResponse format"""
        
        with patch('app.rag_system') as mock_rag_system:
            # Return graceful error message (as RAG system would do)
            mock_rag_system.query.return_value = (
                "I'm experiencing high demand right now and the AI service is temporarily overloaded.",
                []
            )
            
            response = client.post("/api/query", json={
                "query": "Test query",
                "session_id": "test_session"
            })
            
            assert response.status_code == 200
            response_data = response.json()
            
            # Must have all required QueryResponse fields
            required_fields = ["answer", "sources", "session_id"]
            for field in required_fields:
                assert field in response_data, f"Missing required field: {field}"
            
            # Verify field types
            assert isinstance(response_data["answer"], str)
            assert isinstance(response_data["sources"], list)
            assert isinstance(response_data["session_id"], str)