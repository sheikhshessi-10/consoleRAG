"""
Configuration file for USM Brain application.
Centralizes all configurable parameters for easy maintenance.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '150'))
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.5'))

# Sentence Transformer Configuration
MODEL_NAME = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')

# Document Processing Configuration
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '5'))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))

# File Paths
DEFAULT_DOCUMENT_PATH = os.getenv('DEFAULT_DOCUMENT_PATH', 'qp.txt')
FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index.idx')

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
