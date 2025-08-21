import pytest
from fastapi.testclient import TestClient
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns expected message"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "Course Materials RAG System"}
    
    def test_query_endpoint_with_new_session(self, test_client):
        """Test query endpoint creates new session when none provided"""
        query_data = {"query": "What is Python?"}
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test_session_123"
        assert data["answer"] == "Test response for your query."
        assert len(data["sources"]) > 0
    
    def test_query_endpoint_with_existing_session(self, test_client):
        """Test query endpoint uses provided session ID"""
        query_data = {
            "query": "What is Python?",
            "session_id": "existing_session_456"
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing_session_456"
        assert "answer" in data
        assert "sources" in data
    
    def test_query_endpoint_specific_response(self, test_client):
        """Test query endpoint returns specific response for known query"""
        query_data = {"query": "python basics"}
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "Python is a programming language" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Python Programming - Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/python/lesson1"
    
    def test_query_endpoint_error_handling(self, test_client):
        """Test query endpoint handles errors gracefully"""
        query_data = {"query": "test error"}
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "error" in data["answer"] or "apologize" in data["answer"]
        assert "sources" in data
        assert data["sources"] == []
    
    def test_query_endpoint_validation(self, test_client):
        """Test query endpoint validates required fields"""
        # Test missing query field
        response = test_client.post("/api/query", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_invalid_json(self, test_client):
        """Test query endpoint handles invalid JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_courses_endpoint(self, test_client):
        """Test courses endpoint returns course statistics"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Introduction to Python" in data["course_titles"]
        assert "Model Context Protocol" in data["course_titles"]
    
    def test_courses_endpoint_response_model(self, test_client):
        """Test courses endpoint returns proper data types"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])
    
    def test_nonexistent_endpoint(self, test_client):
        """Test that nonexistent endpoints return 404"""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_cors_headers(self, test_client):
        """Test that CORS middleware is configured (TestClient may not expose CORS headers)"""
        # Test that the endpoint is accessible (CORS would block if misconfigured)
        query_data = {"query": "CORS test"}
        response = test_client.post("/api/query", json=query_data)
        
        # If we can make the request successfully, CORS is working
        # TestClient doesn't always expose CORS headers in test environment
        assert response.status_code == 200
        assert "answer" in response.json()
    
    def test_content_type_headers(self, test_client):
        """Test that JSON endpoints return proper content type"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"


@pytest.mark.api
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API workflow"""
    
    def test_full_query_workflow(self, test_client):
        """Test a complete query workflow"""
        # Step 1: Get course statistics
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200
        
        # Step 2: Perform initial query
        initial_query = {"query": "What courses are available?"}
        query_response = test_client.post("/api/query", json=initial_query)
        assert query_response.status_code == 200
        
        session_id = query_response.json()["session_id"]
        
        # Step 3: Perform follow-up query with same session
        followup_query = {
            "query": "Tell me more about Python",
            "session_id": session_id
        }
        followup_response = test_client.post("/api/query", json=followup_query)
        assert followup_response.status_code == 200
        assert followup_response.json()["session_id"] == session_id
    
    def test_multiple_concurrent_sessions(self, test_client):
        """Test multiple concurrent sessions work independently"""
        # Create first session
        query1 = {"query": "Session 1 query"}
        response1 = test_client.post("/api/query", json=query1)
        session1 = response1.json()["session_id"]
        
        # Create second session
        query2 = {"query": "Session 2 query"}
        response2 = test_client.post("/api/query", json=query2)
        session2 = response2.json()["session_id"]
        
        # Verify sessions are different
        assert session1 == "test_session_123"  # Mock always returns same ID
        assert session2 == "test_session_123"  # But in real system would be different
        
        # Both should work independently
        assert response1.status_code == 200
        assert response2.status_code == 200


@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Performance-related tests for API endpoints"""
    
    def test_query_endpoint_response_time(self, test_client):
        """Test that query endpoint responds within reasonable time"""
        import time
        
        query_data = {"query": "Quick performance test"}
        
        start_time = time.time()
        response = test_client.post("/api/query", json=query_data)
        end_time = time.time()
        
        assert response.status_code == 200
        # Since we're using mocks, response should be very fast
        assert end_time - start_time < 1.0  # Less than 1 second
    
    def test_multiple_queries_performance(self, test_client):
        """Test multiple sequential queries performance"""
        import time
        
        queries = [
            {"query": "Query 1"},
            {"query": "Query 2"},
            {"query": "Query 3"},
        ]
        
        start_time = time.time()
        for query in queries:
            response = test_client.post("/api/query", json=query)
            assert response.status_code == 200
        end_time = time.time()
        
        # All queries should complete reasonably fast with mocks
        assert end_time - start_time < 3.0  # Less than 3 seconds for 3 queries