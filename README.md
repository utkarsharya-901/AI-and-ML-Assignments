# RAG Chatbot using LangChain & ChromaDB

This project implements a Retrieval-Augmented Generation (RAG) chatbot pipeline. It allows you to ingest documents (PDFs, TXTs), embed them using OpenAI, store them locally via ChromaDB, and interact with your data conversationally.

## Prerequisites
1. Install Python 3.10+
2. Set your OpenAI API key in your terminal:
   `export OPENAI_API_KEY="your-api-key"` (Mac/Linux)
   `set OPENAI_API_KEY="your-api-key"` (Windows)

## Setup
1. Create a virtual environment and install dependencies:
   `pip install -r requirements.txt`
2. Create a folder named `documents` in the root directory and drop your PDF or TXT files inside.

## Usage
**Step 1: Ingest Data**
Run this command to process your files, generate embeddings, and build the ChromaDB vector database:
`python rag_chatbot.py --ingest`

**Step 2: Chat with your Data**
Run this command to launch the interactive CLI chatbot. It supports conversational memory, so it remembers follow-up questions!
`python rag_chatbot.py --chat`
