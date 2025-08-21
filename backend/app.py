import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

import logging
import os
import traceback
from typing import List, Optional, Union

from config import config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_system import RAGSystem

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with debug mode
app = FastAPI(title="Course Materials RAG System", root_path="", debug=True)

# Add trusted host middleware for proxy
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""

    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    """Model for individual source items that can be either text or clickable links"""

    text: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""

    answer: str
    sources: List[Union[str, SourceItem]]  # Support both legacy strings and new objects
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""

    total_courses: int
    course_titles: List[str]


# API Endpoints


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        logger.info(f"Processing query: {request.query}")

        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()

        # Process query using RAG system (handles API errors gracefully)
        answer, sources = rag_system.query(request.query, session_id)

        logger.info(
            f"Query processed successfully. Sources type: {type(sources)}, Sources: {sources}"
        )

        return QueryResponse(answer=answer, sources=sources, session_id=session_id)
    except Exception as e:
        # Only catch truly unexpected errors (RAG system handles API errors gracefully)
        # This should rarely be reached as RAG system has comprehensive error handling
        logger.error(f"Unexpected error in FastAPI layer: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Return a graceful error response instead of HTTP 500
        return QueryResponse(
            answer="I encountered an unexpected system error. Please try again, and if the problem persists, please contact support.",
            sources=[],
            session_id=session_id or "error_session",
        )


@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    docs_path = "../docs"
    if os.path.exists(docs_path):
        print("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(
                docs_path, clear_existing=False
            )
            print(f"Loaded {courses} courses with {chunks} chunks")
        except Exception as e:
            print(f"Error loading documents: {e}")


# Custom static file handler with no-cache headers for development


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
