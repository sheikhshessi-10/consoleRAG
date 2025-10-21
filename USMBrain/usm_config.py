"""
Configuration file for USM RAG-Anything system.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', None)

# RAG-Anything Configuration
WORKING_DIR = os.getenv('RAG_WORKING_DIR', './usm_rag_storage')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')

# Model Configuration
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
VISION_MODEL = os.getenv('VISION_MODEL', 'gpt-4o')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')

# Processing Configuration
ENABLE_IMAGE_PROCESSING = os.getenv('ENABLE_IMAGE_PROCESSING', 'true').lower() == 'true'
ENABLE_TABLE_PROCESSING = os.getenv('ENABLE_TABLE_PROCESSING', 'true').lower() == 'true'
ENABLE_EQUATION_PROCESSING = os.getenv('ENABLE_EQUATION_PROCESSING', 'true').lower() == 'true'

# Data Configuration
USM_DATA_FILE = os.getenv('USM_DATA_FILE', 'qp.txt')

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
