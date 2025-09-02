"""
Utility functions for USM Brain application.
Contains helper functions for common operations.
"""

import os
import logging
from typing import List, Tuple
import numpy as np

def setup_logging(log_level: str = 'INFO', log_format: str = None) -> logging.Logger:
    """Set up logging configuration for the application."""
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('usmbrain.log')
        ]
    )
    
    return logging.getLogger(__name__)

def validate_file_path(file_path: str) -> bool:
    """Validate if a file path exists and is readable."""
    if not os.path.exists(file_path):
        return False
    if not os.path.isfile(file_path):
        return False
    if not os.access(file_path, os.R_OK):
        return False
    return True

def safe_chunk_text(text: str, chunk_size: int) -> List[str]:
    """Safely chunk text into smaller pieces, preserving word boundaries."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we're not at the end of the text, try to find a word boundary
        if end < len(text):
            # Look for the last space or punctuation before the chunk size
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space + 1
        
        chunks.append(text[start:end].strip())
        start = end
    
    return [chunk for chunk in chunks if chunk]

def format_search_results(indices: np.ndarray, distances: np.ndarray, documents: List[str]) -> List[Tuple[int, float, str]]:
    """Format search results into a more readable structure."""
    results = []
    for idx, (doc_idx, distance) in enumerate(zip(indices, distances)):
        if doc_idx < len(documents):
            results.append((doc_idx, float(distance), documents[doc_idx]))
    return results

def print_search_results(results: List[Tuple[int, float, str]], query: str):
    """Print search results in a formatted way."""
    print(f"\nSearch results for: '{query}'")
    print("-" * 60)
    
    for i, (doc_idx, distance, content) in enumerate(results, 1):
        print(f"\n{i}. Document {doc_idx} (Similarity: {1 - distance:.3f})")
        print(f"Content: {content[:100]}{'...' if len(content) > 100 else ''}")
        print("-" * 40)
