# USM Brain

An intelligent document search and question-answering system that combines sentence transformers, FAISS similarity search, and OpenAI's GPT models to provide accurate responses based on document content.

## Features

- **Document Processing**: Automatically chunks and processes text documents
- **Semantic Search**: Uses sentence transformers for understanding document meaning
- **Fast Retrieval**: FAISS index for efficient similarity search
- **AI-Powered Answers**: OpenAI GPT models generate human-like responses
- **Interactive Interface**: Command-line interface for asking questions

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd USMBrain
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   - Get your API key from [OpenAI](https://platform.openai.com/api-keys)
   - Edit `USMBrain.py` and replace `'your-openai-api-key-here'` with your actual API key

## Usage

1. **Prepare your documents**:
   - Place your text documents in the project directory
   - Update the `load_documents()` call in `main()` to point to your document file

2. **Run the application**:
   ```bash
   python USMBrain.py
   ```

3. **Ask questions**:
   - The system will process your documents and build a search index
   - Type your questions and get AI-powered answers
   - Type 'exit' to quit

## Example

```
USM Brain
Enter your question ('exit' to quit): What is the USM Brain system?

Answer:
The USM Brain is an intelligent document search and question-answering system that uses advanced natural language processing techniques to provide accurate responses based on document content. It combines sentence transformers for semantic understanding, FAISS for efficient similarity search, and OpenAI's GPT models for generating human-like responses.

--------------------------------------------------
```

## Dependencies

- **numpy**: Numerical computing
- **faiss-cpu**: Fast similarity search
- **openai**: OpenAI API integration
- **pandas**: Data manipulation
- **sentence-transformers**: Text embeddings
- **tqdm**: Progress bars
- **torch**: PyTorch backend
- **transformers**: Hugging Face transformers

## Configuration

- **MODEL_NAME**: Change the sentence transformer model in the code
- **chunk_size**: Adjust document chunking size (default: 500 characters)
- **top_k**: Change number of relevant chunks retrieved (default: 5)
- **max_tokens**: Adjust OpenAI response length (default: 150)

## Notes

- The system requires an internet connection for OpenAI API calls
- Processing large documents may take time during initial setup
- Ensure your OpenAI API key has sufficient credits for API calls
