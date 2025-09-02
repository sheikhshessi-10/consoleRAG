# USM Brain Project Structure

This document outlines the complete structure of the USM Brain project and explains the purpose of each file.

## Project Overview
USM Brain is an intelligent document search and question-answering system that combines sentence transformers, FAISS similarity search, and OpenAI's GPT models.

## File Structure

```
USMBrain/
├── USMBrain.py              # Main application file
├── config.py                # Configuration and environment variables
├── utils.py                 # Utility functions and helpers
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installation script
├── test_installation.py     # Installation verification script
├── install.bat              # Windows installation script
├── install.sh               # Unix/Linux/Mac installation script
├── env.example              # Environment variables template
├── qp.txt                   # Sample document for testing
├── README.md                # Project documentation
├── PROJECT_STRUCTURE.md     # This file
├── .gitignore               # Git ignore rules
├── USMBrain.pyproj         # Visual Studio project file
└── USMBrain.sln            # Visual Studio solution file
```

## File Descriptions

### Core Application Files
- **`USMBrain.py`** - Main application with document processing, search, and Q&A functionality
- **`config.py`** - Centralized configuration management using environment variables
- **`utils.py`** - Helper functions for logging, file validation, text processing, and result formatting

### Configuration Files
- **`requirements.txt`** - Lists all Python package dependencies with version requirements
- **`env.example`** - Template for environment variables (copy to `.env` and fill in your values)
- **`.gitignore`** - Specifies which files Git should ignore

### Installation & Setup Files
- **`setup.py`** - Makes the project installable as a Python package
- **`install.bat`** - Windows batch script for automated installation
- **`install.sh`** - Unix/Linux/Mac shell script for automated installation
- **`test_installation.py`** - Verifies that all dependencies are properly installed

### Documentation Files
- **`README.md`** - Comprehensive project documentation and usage instructions
- **`PROJECT_STRUCTURE.md`** - This file explaining the project structure
- **`qp.txt`** - Sample document content for testing the system

### Development Files
- **`USMBrain.pyproj`** - Visual Studio Python project configuration
- **`USMBrain.sln`** - Visual Studio solution file

## Dependencies

### Core Dependencies
- **numpy** - Numerical computing and array operations
- **faiss-cpu** - Fast similarity search and clustering
- **openai** - OpenAI API integration for GPT models
- **pandas** - Data manipulation and analysis
- **sentence-transformers** - Text embeddings and semantic understanding
- **tqdm** - Progress bars for long-running operations
- **torch** - PyTorch backend for deep learning models
- **transformers** - Hugging Face transformers library

### Additional Dependencies
- **scikit-learn** - Machine learning utilities
- **scipy** - Scientific computing functions
- **python-dotenv** - Environment variable management

## Installation Process

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Set up environment variables**: Copy `env.example` to `.env` and fill in your OpenAI API key
4. **Test installation**: Run `python test_installation.py`
5. **Run the application**: Execute `python USMBrain.py`

## Configuration Options

The system can be configured through environment variables or by editing `config.py`:

- **OpenAI settings**: API key, model, max tokens, temperature
- **Document processing**: Chunk size, number of search results
- **Model settings**: Sentence transformer model selection
- **Logging**: Log level and format

## Usage

1. Place your text documents in the project directory
2. Update the document path in `config.py` or set `DEFAULT_DOCUMENT_PATH` environment variable
3. Run the application and ask questions about your documents
4. The system will provide AI-powered answers based on the document content

## Development

- Use `setup.py` for development installation with `pip install -e .`
- Run tests with `python test_installation.py`
- Follow the logging output for debugging
- Check the utility functions in `utils.py` for common operations
