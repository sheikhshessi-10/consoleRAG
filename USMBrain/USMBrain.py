import numpy as np
import faiss
import openai
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # For progress bars
import os
import logging
import json

# Import our custom modules
from config import *
from utils import setup_logging, validate_file_path, safe_chunk_text, smart_chunk_json_document, format_search_results, print_search_results

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging
logger = setup_logging(LOG_LEVEL, LOG_FORMAT)

# Initialize the model and OpenAI API key
model = SentenceTransformer(MODEL_NAME)
openai.api_key = OPENAI_API_KEY

def load_documents(file_path):
    """Load documents from a text file."""
    if not validate_file_path(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist or is not readable.")
    
    logger.info(f"Loading documents from {file_path}")
    with open(file_path, "r", encoding='utf-8') as file:
        return file.readlines()

def parse_json_knowledge_base(file_path):
    """Parse JSON knowledge base and extract meaningful content."""
    if not validate_file_path(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist or is not readable.")
    
    logger.info(f"Loading JSON knowledge base from {file_path}")
    
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
        
        documents = []
        
        # Extract pages from the JSON structure
        if "pages" in data:
            for page in data["pages"]:
                # Create a meaningful document from each page
                doc_parts = []
                
                # Add title if available
                if "title" in page and page["title"]:
                    doc_parts.append(f"Title: {page['title']}")
                
                # Add URL for reference
                if "url" in page and page["url"]:
                    doc_parts.append(f"URL: {page['url']}")
                
                # Add content (main text)
                if "content" in page and page["content"]:
                    # Clean up the content
                    content = page["content"].replace("Document: ", "").replace("=" * 50, "")
                    doc_parts.append(f"Content: {content}")
                
                # Add headings if available
                if "headings" in page and page["headings"]:
                    headings_text = " | ".join([h.get("text", "") for h in page["headings"] if h.get("text")])
                    if headings_text:
                        doc_parts.append(f"Headings: {headings_text}")
                
                # Combine all parts
                if doc_parts:
                    full_doc = "\n\n".join(doc_parts)
                    documents.append(full_doc)
        
        logger.info(f"Extracted {len(documents)} documents from JSON knowledge base")
        return documents
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise ValueError(f"Invalid JSON format in {file_path}")
    except Exception as e:
        logger.error(f"Error processing JSON knowledge base: {e}")
        raise

def preprocess_documents(docs):
    """Preprocess documents by stripping whitespace."""
    logger.info(f"Preprocessing {len(docs)} documents")
    return [doc.strip() for doc in docs if doc.strip()]

def chunk_document(doc, chunk_size=CHUNK_SIZE, is_json=False):
    """Chunk a document into smaller pieces."""
    if is_json:
        return smart_chunk_json_document(doc, chunk_size, overlap=50)
    else:
        return safe_chunk_text(doc, chunk_size)

def create_embeddings(docs, batch_size=32):
    """Create embeddings for the given documents using batch processing to avoid memory issues."""
    logger.info(f"Creating embeddings for {len(docs)} document chunks in batches of {batch_size}")
    
    all_embeddings = []
    total_batches = (len(docs) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(docs), batch_size), desc="Processing batches"):
        batch_docs = docs[i:i + batch_size]
        batch_embeddings = model.encode(batch_docs, show_progress_bar=False, convert_to_tensor=False)
        all_embeddings.append(batch_embeddings)
    
    # Combine all batch embeddings
    return np.vstack(all_embeddings)

def build_faiss_index(embeddings):
    """Build a FAISS index for the embeddings."""
    logger.info("Building FAISS index")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)  # Embeddings are already numpy arrays
    return index

def save_faiss_index(index, filepath="faiss_index.idx"):
    """Save FAISS index to disk."""
    try:
        faiss.write_index(index, filepath)
        logger.info(f"FAISS index saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")
        return False

def load_faiss_index(filepath="faiss_index.idx"):
    """Load FAISS index from disk."""
    try:
        if os.path.exists(filepath):
            index = faiss.read_index(filepath)
            logger.info(f"FAISS index loaded from {filepath}")
            return index
        else:
            logger.info(f"No existing index found at {filepath}")
            return None
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return None

def should_rebuild_index(index_file, document_file):
    """Check if index should be rebuilt based on file modification times."""
    try:
        if not os.path.exists(index_file):
            return True
        
        index_time = os.path.getmtime(index_file)
        doc_time = os.path.getmtime(document_file)
        
        # Rebuild if document is newer than index
        return doc_time > index_time
    except Exception as e:
        logger.warning(f"Could not check file times: {e}")
        return True  # Rebuild to be safe

def search_documents(index, query, top_k=TOP_K_RESULTS, threshold=SIMILARITY_THRESHOLD):
    """Search for the top_k most relevant documents for a given query with similarity filtering."""
    query_embedding = model.encode([query], convert_to_tensor=False)
    D, I = index.search(query_embedding, k=top_k)
    
    # Filter results by similarity threshold
    filtered_indices = []
    filtered_distances = []
    
    for i, (idx, distance) in enumerate(zip(I[0], D[0])):
        # Convert L2 distance to similarity (lower distance = higher similarity)
        similarity = 1 - (distance / 2)  # Normalize L2 distance to [0,1] similarity
        if similarity >= threshold:
            filtered_indices.append(idx)
            filtered_distances.append(distance)
    
    return np.array(filtered_indices), np.array(filtered_distances)

def get_openai_response(question, context):
    """Get a response from OpenAI's API based on the question and context."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai.api_key)
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=OPENAI_TEMPERATURE
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Error getting response from OpenAI: {e}"

def answer_question(question, context):
    """Combine context and get an answer to the question."""
    combined_context = " ".join(context)
    return get_openai_response(question, combined_context)

def main():
    """Main function to run the USM Brain application."""
    print("USM Brain - Intelligent Document Search & Q&A")
    print("=" * 60)
    
    try:
        # Try to parse as JSON knowledge base first
        if DEFAULT_DOCUMENT_PATH.endswith('.txt'):
            try:
                documents = parse_json_knowledge_base(DEFAULT_DOCUMENT_PATH)
                print(f"Successfully loaded JSON knowledge base with {len(documents)} pages")
            except (json.JSONDecodeError, ValueError):
                # Fallback to regular text file
                documents = load_documents(DEFAULT_DOCUMENT_PATH)
                print(f"Loaded as regular text file with {len(documents)} lines")
        else:
            documents = load_documents(DEFAULT_DOCUMENT_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure the file '{DEFAULT_DOCUMENT_PATH}' exists in the current directory.")
        return

    processed_docs = preprocess_documents(documents)
    
    # Use JSON chunking if we loaded from JSON knowledge base
    is_json_kb = DEFAULT_DOCUMENT_PATH.endswith('.txt') and len(documents) > 0 and any('Title:' in doc for doc in documents)
    
    if is_json_kb:
        print("Using smart JSON chunking for knowledge base...")
        chunked_documents = [chunk for doc in processed_docs for chunk in chunk_document(doc, is_json=True)]
    else:
        chunked_documents = [chunk for doc in processed_docs for chunk in chunk_document(doc)]
    
    if not chunked_documents:
        print("No document content found to process.")
        return
    
    # Check if we need to rebuild the index
    if should_rebuild_index(FAISS_INDEX_PATH, DEFAULT_DOCUMENT_PATH):
        print(f"Document changed or no index found. Processing {len(chunked_documents)} document chunks...")
        embeddings = create_embeddings(chunked_documents, batch_size=EMBEDDING_BATCH_SIZE)
        index = build_faiss_index(embeddings)
        
        # Save index for future use
        if save_faiss_index(index, FAISS_INDEX_PATH):
            print("Index saved for faster future startup!")
        else:
            print("Warning: Could not save index, will rebuild next time")
    else:
        # Load existing index
        index = load_faiss_index(FAISS_INDEX_PATH)
        if index is not None:
            print("Loaded existing FAISS index - much faster startup!")
        else:
            print("Failed to load index, rebuilding...")
            embeddings = create_embeddings(chunked_documents, batch_size=EMBEDDING_BATCH_SIZE)
            index = build_faiss_index(embeddings)
            save_faiss_index(index, FAISS_INDEX_PATH)
    
    print(f"\nReady! You can now ask questions about your documents.")
    print("Type 'exit' to quit the application.")
    print("-" * 60)

    while True:
        try:
            query = input("\nEnter your question: ").strip()
            if not query:
                continue
                
            if query.lower() == 'exit':
                print("Goodbye!")
                break

            print("Searching for relevant information...")
            indices, distances = search_documents(index, query)
            
            if not indices.size:
                print("No relevant documents found with sufficient similarity.")
                print("Try rephrasing your question or using different keywords.")
                continue

            context = [chunked_documents[i] for i in indices]
            
            # Show search results
            results = format_search_results(indices, distances, chunked_documents)
            print_search_results(results, query)
            
            # Show the top 2-3 chunks that will be used for context
            print(f"\nUsing {min(3, len(context))} most relevant chunks for context:")
            print("-" * 60)
            for i, chunk in enumerate(context[:3], 1):
                print(f"\nContext {i}:")
                print(f"{chunk[:200]}{'...' if len(chunk) > 200 else ''}")
                print("-" * 40)
            
            print("\nGenerating answer...")
            answer = answer_question(query, context)
            print(f"\nAnswer:\n{answer}")
            print("\n" + "-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
