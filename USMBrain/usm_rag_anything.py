#!/usr/bin/env python3
"""
USM RAG-Anything: Advanced Multimodal RAG System for University of Southern Mississippi
Replaces the simple USM Brain with RAG-Anything for enhanced capabilities.
"""

import asyncio
import os
import json
import logging
from typing import List, Dict, Any
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class USMRAGAnything:
    """Advanced RAG system for USM using RAG-Anything framework."""
    
    def __init__(self, api_key: str, base_url: str = None):
        """Initialize the USM RAG-Anything system."""
        self.api_key = api_key
        self.base_url = base_url
        self.rag = None
        self.initialized = False
        
    async def initialize(self, working_dir: str = "./usm_rag_storage"):
        """Initialize the RAG-Anything system."""
        try:
            # Create RAGAnything configuration
            config = RAGAnythingConfig(
                working_dir=working_dir,
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )
            
            # Define LLM model function
            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return openai_complete_if_cache(
                    "gpt-4o-mini",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )
            
            # Define vision model function for image processing
            def vision_model_func(
                prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
            ):
                # If messages format is provided (for multimodal VLM enhanced query), use it directly
                if messages:
                    return openai_complete_if_cache(
                        "gpt-4o",
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=messages,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        **kwargs,
                    )
                # Traditional single image format
                elif image_data:
                    return openai_complete_if_cache(
                        "gpt-4o",
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=[
                            {"role": "system", "content": system_prompt}
                            if system_prompt
                            else None,
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        },
                                    },
                                ],
                            }
                            if image_data
                            else {"role": "user", "content": prompt},
                        ],
                        api_key=self.api_key,
                        base_url=self.base_url,
                        **kwargs,
                    )
                # Pure text format
                else:
                    return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
            
            # Define embedding function
            embedding_func = EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-3-large",
                    api_key=self.api_key,
                    base_url=self.base_url,
                ),
            )
            
            # Initialize RAGAnything
            self.rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
            )
            
            # Initialize the LightRAG instance
            await self.rag.lightrag.initialize_storages()
            
            self.initialized = True
            logger.info("USM RAG-Anything system initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG-Anything: {e}")
            raise
    
    async def load_usm_data(self, json_file_path: str = "qp.txt"):
        """Load and process USM data from JSON file."""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            logger.info(f"Loading USM data from {json_file_path}")
            
            # Read the JSON file
            with open(json_file_path, "r", encoding='utf-8') as file:
                data = json.load(file)
            
            # Convert JSON data to content list format
            content_list = []
            
            if "pages" in data:
                for page in data["pages"]:
                    # Create text content for each page
                    page_content = []
                    
                    # Add title if available
                    if "title" in page and page["title"]:
                        page_content.append(f"Title: {page['title']}")
                    
                    # Add URL for reference
                    if "url" in page and page["url"]:
                        page_content.append(f"URL: {page['url']}")
                    
                    # Add content (main text)
                    if "content" in page and page["content"]:
                        content = page["content"].replace("Document: ", "").replace("=" * 50, "")
                        page_content.append(f"Content: {content}")
                    
                    # Add headings if available
                    if "headings" in page and page["headings"]:
                        headings_text = " | ".join([h.get("text", "") for h in page["headings"] if h.get("text")])
                        if headings_text:
                            page_content.append(f"Headings: {headings_text}")
                    
                    # Combine all parts
                    if page_content:
                        full_content = "\n\n".join(page_content)
                        content_list.append({
                            "type": "text",
                            "text": full_content,
                            "page_idx": len(content_list)  # Use index as page number
                        })
            
            logger.info(f"Converted {len(content_list)} pages to content list")
            
            # Insert the content list directly into RAG-Anything
            await self.rag.insert_content_list(
                content_list=content_list,
                file_path="usm_website_data.json",
                display_stats=True
            )
            
            logger.info("USM data successfully loaded into RAG-Anything!")
            return len(content_list)
            
        except Exception as e:
            logger.error(f"Failed to load USM data: {e}")
            raise
    
    async def query(self, question: str, mode: str = "hybrid", vlm_enhanced: bool = True):
        """Query the USM knowledge base."""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            # Use VLM enhanced query if vision model is available
            result = await self.rag.aquery(
                question,
                mode=mode,
                vlm_enhanced=vlm_enhanced
            )
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Error processing query: {e}"
    
    async def query_with_multimodal(self, question: str, multimodal_content: List[Dict], mode: str = "hybrid"):
        """Query with specific multimodal content."""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        try:
            result = await self.rag.aquery_with_multimodal(
                question,
                multimodal_content=multimodal_content,
                mode=mode
            )
            return result
        except Exception as e:
            logger.error(f"Multimodal query failed: {e}")
            return f"Error processing multimodal query: {e}"

async def main():
    """Main function to run the USM RAG-Anything application."""
    print("USM RAG-Anything - Advanced Multimodal RAG System")
    print("=" * 60)
    
    # Get API key from environment or user input
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("ERROR: API key is required!")
            return
    
    # Initialize the system
    usm_rag = USMRAGAnything(api_key=api_key)
    
    try:
        print("Initializing RAG-Anything system...")
        await usm_rag.initialize()
        
        print("Loading USM knowledge base...")
        pages_loaded = await usm_rag.load_usm_data("qp.txt")
        print(f"SUCCESS: Loaded {pages_loaded} pages from USM website data")
        
        print("\nReady! You can now ask questions about USM.")
        print("Type 'exit' to quit the application.")
        print("-" * 60)
        
        # Interactive query loop
        while True:
            try:
                question = input("\nEnter your question: ").strip()
                if not question:
                    continue
                    
                if question.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                print("Searching for relevant information...")
                answer = await usm_rag.query(question)
                print(f"\nAnswer:\n{answer}")
                print("\n" + "-" * 60)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"ERROR: An error occurred: {e}")
    
    except Exception as e:
        logger.error(f"Failed to start USM RAG-Anything: {e}")
        print(f"ERROR: Failed to start system: {e}")

if __name__ == "__main__":
    asyncio.run(main())
