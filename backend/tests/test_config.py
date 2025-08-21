import os
from dataclasses import dataclass

import pytest
from config import Config


class TestConfig:
    """Test suite for configuration validation"""

    def test_config_validation_success(self):
        """Test that valid configuration passes validation"""
        # This should work with default values
        config = Config()
        assert config.MAX_RESULTS == 5
        assert config.CHUNK_SIZE == 800
        assert config.CHUNK_OVERLAP == 100

    def test_max_results_validation(self):
        """Test MAX_RESULTS validation"""

        # Create config class with invalid MAX_RESULTS
        @dataclass
        class InvalidConfig(Config):
            MAX_RESULTS: int = 0

        with pytest.raises(ValueError, match="MAX_RESULTS must be > 0"):
            InvalidConfig()

    def test_chunk_size_validation(self):
        """Test CHUNK_SIZE validation"""

        @dataclass
        class InvalidConfig(Config):
            CHUNK_SIZE: int = 50

        with pytest.raises(ValueError, match="CHUNK_SIZE must be >= 100"):
            InvalidConfig()

    def test_chunk_overlap_validation(self):
        """Test CHUNK_OVERLAP validation"""

        # Test negative overlap
        @dataclass
        class InvalidConfig1(Config):
            CHUNK_OVERLAP: int = -10

        with pytest.raises(ValueError, match="CHUNK_OVERLAP must be 0"):
            InvalidConfig1()

        # Test overlap >= chunk_size
        @dataclass
        class InvalidConfig2(Config):
            CHUNK_SIZE: int = 100
            CHUNK_OVERLAP: int = 100

        with pytest.raises(ValueError, match="CHUNK_OVERLAP must be 0"):
            InvalidConfig2()

    def test_max_retries_validation(self):
        """Test MAX_RETRIES validation"""

        @dataclass
        class InvalidConfig(Config):
            MAX_RETRIES: int = -1

        with pytest.raises(ValueError, match="MAX_RETRIES must be >= 0"):
            InvalidConfig()

    def test_api_key_validation(self, monkeypatch):
        """Test ANTHROPIC_API_KEY validation"""
        # Mock empty API key
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        @dataclass
        class TestConfig(Config):
            ANTHROPIC_API_KEY: str = ""

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
            TestConfig()

    def test_retry_configuration_defaults(self):
        """Test retry configuration has reasonable defaults"""
        config = Config()
        assert config.MAX_RETRIES == 3
        assert config.RETRY_DELAY == 1.0
        assert config.MAX_RETRY_DELAY == 60.0

    def test_valid_edge_cases(self):
        """Test valid edge cases pass validation"""

        @dataclass
        class EdgeCaseConfig(Config):
            ANTHROPIC_API_KEY: str = "test_key"
            CHUNK_SIZE: int = 100  # Minimum valid size
            CHUNK_OVERLAP: int = 0  # Minimum valid overlap
            MAX_RESULTS: int = 1  # Minimum valid results
            MAX_RETRIES: int = 0  # Minimum valid retries

        # Should not raise any exceptions
        config = EdgeCaseConfig()
        assert config.CHUNK_SIZE == 100
        assert config.CHUNK_OVERLAP == 0
        assert config.MAX_RESULTS == 1
        assert config.MAX_RETRIES == 0
