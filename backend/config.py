import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    
    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location
    
    # API retry settings
    MAX_RETRIES: int = 3         # Maximum number of API retries
    RETRY_DELAY: float = 1.0     # Base delay between retries (seconds)
    MAX_RETRY_DELAY: float = 60.0  # Maximum delay between retries
    
    def __post_init__(self):
        """Validate configuration values"""
        if not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required")
        
        if self.MAX_RESULTS < 1:
            raise ValueError(f"MAX_RESULTS must be > 0, got {self.MAX_RESULTS}")
        
        if self.CHUNK_SIZE < 100:
            raise ValueError(f"CHUNK_SIZE must be >= 100, got {self.CHUNK_SIZE}")
        
        if self.CHUNK_OVERLAP < 0 or self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(f"CHUNK_OVERLAP must be 0 <= overlap < chunk_size, got {self.CHUNK_OVERLAP}")
        
        if self.MAX_RETRIES < 0:
            raise ValueError(f"MAX_RETRIES must be >= 0, got {self.MAX_RETRIES}")

config = Config()


