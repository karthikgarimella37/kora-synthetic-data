"""
Configuration file for the Response Generator system.
Modify these settings to customize the behavior.
"""

import os
from typing import Dict, Any

# =============================================================================
# LLM Backend Configuration
# =============================================================================

# Available backends: "openai", "gpt4all", "huggingface"
DEFAULT_LLM_BACKEND = "openai"

# OpenAI Configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),  # Set via environment variable
    "model": "gpt-3.5-turbo",  # Options: gpt-3.5-turbo, gpt-4, gpt-4-turbo
    "temperature": 0.7,
    "max_tokens": 500
}

# GPT4All Configuration (for local models)
GPT4ALL_CONFIG = {
    "model_path": "orca-mini-3b.ggmlv3.q4_0.bin",  # Path to downloaded model
    "temperature": 0.7,
    "max_tokens": 300
}

# HuggingFace Configuration (for local transformers)
HUGGINGFACE_CONFIG = {
    "model_name": "microsoft/DialoGPT-medium",  # Model from HF Hub
    "device": "auto",  # "auto", "cpu", "cuda"
    "temperature": 0.8,
    "max_tokens": 400
}

# =============================================================================
# Embedding and Index Configuration
# =============================================================================

EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",  # HuggingFace embedding model
    "index_path": "faiss_index.index",
    "text_chunks_path": "text_chunks.pkl",
    "max_context_length": 4000  # Max characters in context
}

# =============================================================================
# Search Configuration
# =============================================================================

SEARCH_CONFIG = {
    "default_k": 5,  # Number of similar chunks to retrieve
    "max_k": 20,     # Maximum number of chunks allowed
    "similarity_threshold": 0.0  # Minimum similarity score (if supported)
}

# =============================================================================
# Response Generation Configuration
# =============================================================================

RESPONSE_CONFIG = {
    "default_system_prompt": """You are a helpful AI assistant that answers questions based on the provided context. 
Use the context information to provide accurate and relevant answers. If the context doesn't contain 
enough information to answer the question, say so clearly. Always cite the relevant chunks/pages when possible.""",
    
    "academic_system_prompt": """You are an academic research assistant. Provide scholarly, well-referenced answers 
based on the provided context. Use formal academic language and cite specific sections when making claims.""",
    
    "simple_system_prompt": """You are explaining complex topics to a general audience. Use simple, clear language 
and avoid technical jargon. Explain concepts in an accessible way while staying accurate to the source material.""",
    
    "technical_system_prompt": """You are a technical expert providing detailed technical analysis. Focus on 
methodology, implementation details, and technical specifications found in the context."""
}

# =============================================================================
# File Paths and Directories
# =============================================================================

PATHS = {
    "data_dir": "data",
    "models_dir": "models",
    "output_dir": "output",
    "logs_dir": "logs"
}

# =============================================================================
# Logging Configuration
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(PATHS["logs_dir"], "response_generator.log")
}

# =============================================================================
# Performance Configuration
# =============================================================================

PERFORMANCE_CONFIG = {
    "batch_size": 10,  # For batch processing
    "timeout_seconds": 60,  # Request timeout
    "max_retries": 3,  # Number of retries on failure
    "cache_responses": True  # Cache responses to avoid re-computation
}

# =============================================================================
# UI Configuration (for interactive mode)
# =============================================================================

UI_CONFIG = {
    "show_sources_by_default": False,
    "show_timing_info": True,
    "show_chunk_count": True,
    "max_display_length": 200  # Max characters to display per chunk
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_config_for_backend(backend: str) -> Dict[str, Any]:
    """Get configuration for a specific LLM backend."""
    backend_configs = {
        "openai": OPENAI_CONFIG,
        "gpt4all": GPT4ALL_CONFIG,
        "huggingface": HUGGINGFACE_CONFIG
    }
    
    if backend not in backend_configs:
        raise ValueError(f"Unknown backend: {backend}")
    
    return backend_configs[backend]


def validate_config() -> Dict[str, str]:
    """Validate configuration and return any issues found."""
    issues = {}
    
    # Check OpenAI API key if using OpenAI backend
    if DEFAULT_LLM_BACKEND == "openai" and not OPENAI_CONFIG["api_key"]:
        issues["openai_api_key"] = "OPENAI_API_KEY environment variable not set"
    
    # Check if required files exist
    if not os.path.exists(EMBEDDING_CONFIG["index_path"]):
        issues["index_file"] = f"Index file not found: {EMBEDDING_CONFIG['index_path']}"
    
    if not os.path.exists(EMBEDDING_CONFIG["text_chunks_path"]):
        issues["chunks_file"] = f"Chunks file not found: {EMBEDDING_CONFIG['text_chunks_path']}"
    
    # Check directories
    for dir_name, dir_path in PATHS.items():
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                issues[f"directory_{dir_name}"] = f"Cannot create directory {dir_path}: {e}"
    
    return issues


def print_config_summary():
    """Print a summary of the current configuration."""
    print("Response Generator Configuration Summary")
    print("=" * 50)
    print(f"Default LLM Backend: {DEFAULT_LLM_BACKEND}")
    print(f"Embedding Model: {EMBEDDING_CONFIG['model_name']}")
    print(f"Default K (chunks): {SEARCH_CONFIG['default_k']}")
    print(f"Max Context Length: {EMBEDDING_CONFIG['max_context_length']}")
    
    if DEFAULT_LLM_BACKEND == "openai":
        print(f"OpenAI Model: {OPENAI_CONFIG['model']}")
        print(f"OpenAI API Key: {'✓ Set' if OPENAI_CONFIG['api_key'] else '✗ Not set'}")
    
    # Check for issues
    issues = validate_config()
    if issues:
        print("\n⚠️  Configuration Issues:")
        for key, issue in issues.items():
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration is valid")


# =============================================================================
# Environment Setup
# =============================================================================

def setup_environment():
    """Set up the environment with required directories."""
    for dir_path in PATHS.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print("Environment setup complete")


if __name__ == "__main__":
    print_config_summary()
    
    # Optionally set up environment
    setup_choice = input("\nSet up directories? (y/n): ").strip().lower()
    if setup_choice == 'y':
        setup_environment()
