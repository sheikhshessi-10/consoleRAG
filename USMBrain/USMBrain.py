import numpy as np
import faiss
import openai
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # For progress bars
import os
import logging

# Import our custom modules
from config import *
from utils import setup_logging, validate_file_path, safe_chunk_text, format_search_results, print_search_results

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

def preprocess_documents(docs):
    """Preprocess documents by stripping whitespace."""
    logger.info(f"Preprocessing {len(docs)} documents")
    return [doc.strip() for doc in docs if doc.strip()]

def chunk_document(doc, chunk_size=CHUNK_SIZE):
    """Chunk a document into smaller pieces."""
    return safe_chunk_text(doc, chunk_size)

def create_embeddings(docs):
    """Create embeddings for the given documents."""
    logger.info(f"Creating embeddings for {len(docs)} document chunks")
    return model.encode(docs, show_progress_bar=True, convert_to_tensor=True)

def build_faiss_index(embeddings):
    """Build a FAISS index for the embeddings."""
    logger.info("Building FAISS index")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())  # Ensure embeddings are on CPU
    return index

def search_documents(index, query, top_k=TOP_K_RESULTS):
    """Search for the top_k most relevant documents for a given query."""
    query_embedding = model.encode([query], convert_to_tensor=True)
    D, I = index.search(query_embedding.cpu().numpy(), k=top_k)
    return I[0], D[0]  # Return the indices and distances of the top documents

def get_openai_response(question, context):
    """Get a response from OpenAI's API based on the question and context."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=OPENAI_TEMPERATURE
        )
        
        return response.choices[0].message['content'].strip()
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
        documents = load_documents(DEFAULT_DOCUMENT_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure the file '{DEFAULT_DOCUMENT_PATH}' exists in the current directory.")
        return

    processed_docs = preprocess_documents(documents)
    chunked_documents = [chunk for doc in processed_docs for chunk in chunk_document(doc)]
    
    if not chunked_documents:
        print("No document content found to process.")
        return
    
    print(f"Processing {len(chunked_documents)} document chunks...")
    embeddings = create_embeddings(chunked_documents)
    index = build_faiss_index(embeddings)
    
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
                print("No relevant documents found.")
                continue

            context = [chunked_documents[i] for i in indices]
            
            # Show search results
            results = format_search_results(indices, distances, chunked_documents)
            print_search_results(results, query)
            
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
