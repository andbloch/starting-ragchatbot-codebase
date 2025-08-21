# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Quick start**: `chmod +x run.sh && ./run.sh`
- **Manual start**: `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Install dependencies**: `uv sync`

### Code Quality
- **Format code**: `./scripts/format.sh` (Black + isort)
- **Lint code**: `./scripts/lint.sh` (flake8, mypy, format checks)
- **Full quality check**: `./scripts/quality.sh` (format + lint)

### Environment Setup
- Create `.env` file in root with `ANTHROPIC_API_KEY=your_key_here`
- Requires Python 3.13+, uv package manager, and Anthropic API key

### Application URLs
- Web interface: http://localhost:8000
- API docs: http://localhost:8000/docs

## Architecture Overview

This is a full-stack RAG (Retrieval-Augmented Generation) system for querying course materials:

### Backend Architecture (`backend/`)
- **FastAPI application** (`app.py`): Main web server with CORS, static file serving, and API endpoints
- **RAG System** (`rag_system.py`): Core orchestrator managing all components
- **Document Processing** (`document_processor.py`): Handles parsing and chunking of course materials
- **Vector Store** (`vector_store.py`): ChromaDB integration for semantic search
- **AI Generator** (`ai_generator.py`): Anthropic Claude integration with tool support
- **Session Management** (`session_manager.py`): Conversation history tracking
- **Search Tools** (`search_tools.py`): Tool-based search functionality for Claude
- **Models** (`models.py`): Data models for courses, lessons, and chunks
- **Configuration** (`config.py`): Centralized configuration management

### Frontend (`frontend/`)
- **HTML Structure** (`index.html`): Single-page application with sidebar and chat interface
- **JavaScript Logic** (`script.js`): Event handling, API communication, session management, markdown rendering
- **CSS Styling** (`style.css`): Dark theme UI with responsive design and animations

### Frontend Architecture Diagram

```mermaid
graph TB
    %% External dependencies
    Browser[Web Browser]
    MarkedJS[Marked.js Library<br/>Markdown rendering]
    
    %% Main HTML structure
    HTML[index.html<br/>Main Document]
    
    %% Core components
    JS[script.js<br/>Application Logic]
    CSS[style.css<br/>Styling & Layout]
    
    %% UI Components
    Header[Header Section<br/>Title & subtitle]
    Sidebar[Sidebar Component<br/>Stats & suggestions]
    ChatMain[Chat Main Area<br/>Messages & input]
    
    %% Sidebar sub-components
    CourseStats[Course Statistics<br/>Total courses & titles]
    SuggestedQuestions[Suggested Questions<br/>Clickable prompts]
    
    %% Chat sub-components
    ChatMessages[Chat Messages<br/>Conversation history]
    ChatInput[Chat Input<br/>Text input & send button]
    
    %% JavaScript modules
    EventHandlers[Event Handlers<br/>User interactions]
    APIClient[API Client<br/>Backend communication]
    SessionManager[Session Manager<br/>Conversation state]
    UIUpdater[UI Updater<br/>DOM manipulation]
    
    %% Data flow
    Browser -->|Loads| HTML
    HTML -->|References| CSS
    HTML -->|References| JS
    HTML -->|References| MarkedJS
    
    HTML --> Header
    HTML --> Sidebar
    HTML --> ChatMain
    
    Sidebar --> CourseStats
    Sidebar --> SuggestedQuestions
    
    ChatMain --> ChatMessages
    ChatMain --> ChatInput
    
    JS --> EventHandlers
    JS --> APIClient
    JS --> SessionManager
    JS --> UIUpdater
    
    EventHandlers -->|User clicks| SuggestedQuestions
    EventHandlers -->|User types| ChatInput
    APIClient -->|Fetches| CourseStats
    APIClient -->|Posts queries| ChatMessages
    SessionManager -->|Maintains| ChatMessages
    UIUpdater -->|Updates| ChatMessages
    UIUpdater -->|Updates| CourseStats
    
    MarkedJS -->|Renders| ChatMessages
    
    %% Styling
    classDef external fill:#e1f5fe
    classDef structure fill:#f3e5f5
    classDef component fill:#e8f5e8
    classDef logic fill:#fff3e0
    
    class Browser,MarkedJS external
    class HTML,CSS structure
    class Header,Sidebar,ChatMain,CourseStats,SuggestedQuestions,ChatMessages,ChatInput component
    class JS,EventHandlers,APIClient,SessionManager,UIUpdater logic
```

### Key Design Patterns
- **Tool-based AI**: Uses Claude's tool-calling capability for structured searches
- **Session-based conversations**: Maintains context across queries
- **Modular architecture**: Clear separation between processing, storage, AI, and web layers
- **Chunked processing**: Documents split into semantic chunks for better retrieval

### Backend Architecture Diagram

```mermaid
graph TB
    %% External components
    Client[Web Client]
    Docs[docs/ folder<br/>PDF/DOCX/TXT files]
    Claude[Anthropic Claude API]
    
    %% Main FastAPI app
    App[FastAPI App<br/>app.py]
    
    %% Core RAG system
    RAG[RAG System<br/>rag_system.py<br/>Main Orchestrator]
    
    %% Processing components
    DocProc[Document Processor<br/>document_processor.py<br/>Parse & chunk documents]
    VectorStore[Vector Store<br/>vector_store.py<br/>ChromaDB integration]
    AIGen[AI Generator<br/>ai_generator.py<br/>Claude integration]
    SessionMgr[Session Manager<br/>session_manager.py<br/>Conversation history]
    SearchTools[Search Tools<br/>search_tools.py<br/>Tool definitions]
    
    %% Data models
    Models[Models<br/>models.py<br/>Course/Lesson/Chunk]
    Config[Configuration<br/>config.py<br/>Settings & env vars]
    
    %% Data flow
    Client -->|HTTP requests| App
    App -->|Query processing| RAG
    Docs -->|Startup load| RAG
    
    RAG -->|Orchestrates| DocProc
    RAG -->|Orchestrates| VectorStore
    RAG -->|Orchestrates| AIGen
    RAG -->|Orchestrates| SessionMgr
    RAG -->|Orchestrates| SearchTools
    
    DocProc -->|Uses| Models
    DocProc -->|Chunks to| VectorStore
    AIGen -->|API calls| Claude
    AIGen -->|Uses tools| SearchTools
    SearchTools -->|Searches| VectorStore
    
    Config -->|Configures| RAG
    Config -->|Configures| VectorStore
    Config -->|Configures| AIGen
    
    %% Styling
    classDef external fill:#e1f5fe
    classDef core fill:#f3e5f5
    classDef component fill:#e8f5e8
    classDef data fill:#fff3e0
    
    class Client,Docs,Claude external
    class App,RAG core
    class DocProc,VectorStore,AIGen,SessionMgr,SearchTools component
    class Models,Config data
```

### Data Flow
1. Course documents (PDF/DOCX/TXT) loaded from `docs/` folder on startup
2. Documents processed into Course/Lesson objects and text chunks
3. Content stored in ChromaDB with embeddings via sentence-transformers
4. User queries processed through RAG pipeline with session context
5. Claude generates responses using search tools to find relevant content

### Configuration
- All settings centralized in `config.py`
- Environment variables loaded via python-dotenv
- ChromaDB persistence path, chunk sizes, and model settings configurable