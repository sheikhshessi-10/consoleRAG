#!/usr/bin/env python3
"""
USM LightRAG: Direct LightRAG implementation for USM
"""

import asyncio
import os
import json
import logging
from typing import List, Dict, Any
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the USM LightRAG application."""
    print("USM LightRAG - Advanced RAG System")
    print("=" * 60)
    
    # Get API key from environment or user input
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("ERROR: API key is required!")
            return
    
    try:
        print("Initializing LightRAG system...")
        
        # Initialize LightRAG
        rag = LightRAG(
            working_dir="./usm_lightrag_storage",
            llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                **kwargs,
            ),
            embedding_func=EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-3-large",
                    api_key=api_key,
                ),
            )
        )
        
        # Initialize storage
        await rag.initialize_storages()
        await initialize_pipeline_status()
        
        print("Loading USM knowledge base...")
        
        # Read the JSON file
        with open("qp.txt", "r", encoding='utf-8') as file:
            data = json.load(file)
        
        # Process each page
        pages_processed = 0
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
                    
                    # Insert into LightRAG
                    await rag.ainsert(
                        full_content,
                        file_paths=["usm_website_data.json"]
                    )
                    pages_processed += 1
                    
                    if pages_processed % 50 == 0:
                        print(f"Processed {pages_processed} pages...")
        
        print(f"SUCCESS: Loaded {pages_processed} pages from USM website data")
        
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
                answer = await rag.aquery(question, mode="hybrid")
                print(f"\nAnswer:\n{answer}")
                print("\n" + "-" * 60)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"ERROR: An error occurred: {e}")
    
    except Exception as e:
        logger.error(f"Failed to start USM LightRAG: {e}")
        print(f"ERROR: Failed to start system: {e}")

if __name__ == "__main__":
    asyncio.run(main())
