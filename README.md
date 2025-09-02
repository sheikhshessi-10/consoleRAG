# Console RAG - RAG Pipeline for Intelligent Document Q&A

A complete **Retrieval-Augmented Generation (RAG) pipeline** that transforms your documents into an intelligent knowledge base. Combines semantic search with AI generation to provide accurate, context-aware answers from your content.

## ✨ Features

- 🔍 **Semantic Document Search** - Advanced NLP-powered document understanding
- 🤖 **AI-Powered Q&A** - OpenAI GPT integration for intelligent responses
- ⚡ **Fast Vector Search** - FAISS indexing for lightning-fast retrieval
- 📄 **Smart Text Chunking** - Preserves word boundaries for better context
- 🛠️ **Easy Configuration** - Environment-based settings management
- 📊 **Comprehensive Logging** - Detailed logging for debugging and monitoring
- 🚀 **Cross-Platform** - Windows, Linux, and macOS support
- 🔧 **Production Ready** - Robust error handling and validation

## 🏗️ RAG Architecture

```
Documents → Chunking → Embeddings → Vector Index → Query → Retrieval → Context → AI Generation → Answer
```

1. **Document Processing**: Intelligent chunking with word boundary preservation
2. **Semantic Embeddings**: Sentence transformers for understanding document meaning
3. **Vector Search**: FAISS index for fast similarity search
4. **Context Retrieval**: Top-k most relevant document chunks
5. **AI Generation**: OpenAI GPT generates answers based on retrieved context

