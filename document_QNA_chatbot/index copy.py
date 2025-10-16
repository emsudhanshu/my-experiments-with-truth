# This script implements a simple Retrieval-Augmented Generation (RAG) system
# for a Proof of Concept (PoC) using LlamaIndex and Ollama.
#
# PREREQUISITES:
# 1. Ollama server must be running locally (usually on port 11434).
# 2. The required models must be pulled:
#    ollama pull llama3
#    ollama pull nomic-embed-text
# 3. Install Python dependencies:
#    pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama pydantic
#    pip install pypdf llama-index-readers-file  <-- NEW DEPENDENCIES

import os
import sys
# Import the SimpleDirectoryReader for file loading
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama as OllamaLLM
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader

# --- Configuration ---
# Ollama runs on http://localhost:11434 by default.
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3" # LLM for generation
EMBEDDING_MODEL = "nomic-embed-text" # Embedding model for vector creation

# Path to the PDF file to be loaded
# NOTE: This assumes the file is accessible at this path relative to the script
PDF_FILE_PATH = "project_document.pdf" 

# --- Document Content (REMOVED: Now loading from PDF) ---

def initialize_rag_system():
    """Initializes the Ollama models and creates the VectorStoreIndex."""
    print("--- 1. Initializing RAG System ---")

    # 1. Setup Ollama LLM and Embedding Model
    try:
        llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        print(f"✅ Ollama LLM ({LLM_MODEL}) and Embedding ({EMBEDDING_MODEL}) configured.")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama. Is the server running at {OLLAMA_BASE_URL}? (Error: {e})")
        # Check for model-specific errors
        if "model" in str(e) and "not found" in str(e):
             print(f"HINT: Please ensure you ran 'ollama pull {LLM_MODEL}' and 'ollama pull {EMBEDDING_MODEL}' in your terminal.")
        sys.exit(1)

    # 2. Load the Document from PDF
    print(f"--- 2. Loading Document from {PDF_FILE_PATH} ---")
    try:
        # Load the document using SimpleDirectoryReader
        # We target the specific file path instead of a full directory scan for simplicity
        documents = SimpleDirectoryReader(input_files=[PDF_FILE_PATH]).load_data()
        print(f"✅ Document loaded successfully. Found {len(documents)} text chunks/pages.")
    except Exception as e:
        print(f"❌ Failed to load document from {PDF_FILE_PATH}. (Error: {e})")
        print("HINT: Ensure the 'project_document.pdf' file exists in the same directory as the script, and you have installed 'pypdf' and 'llama-index-readers-file'.")
        sys.exit(1)


    # 3. Create Index and Vector Database
    print("--- 3. Creating Index and Vector Database ---")

    # The index creation handles chunking, embedding generation, and storing vectors.
    try:
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            show_progress=True
        )
        print("✅ Index created successfully. Document embeddings stored.")
    except Exception as e:
        print(f"❌ Failed to create the index. The issue is likely with the Ollama server instability during embedding generation. (Error: {e})")
        print("HINT: Try restarting your Ollama server and ensuring it has adequate resources (especially RAM) to load the embedding model.")
        sys.exit(1)

    # 4. Create the Query Engine (RAG pipeline)
    query_engine = index.as_query_engine(
        llm=llm,
        streaming=False, # Set to True for streaming responses
        similarity_top_k=3 # Retrieve top 3 relevant chunks
    )
    print("✅ Query Engine is ready.")

    return query_engine

def chat_loop(query_engine):
    """Starts the interactive chat loop."""
    print("\n" * 2)
    print("==============================================")
    print(f"🤖 RAG Chatbot Proof of Concept (Ollama/{LLM_MODEL})")
    print("==============================================")
    print("Ask me anything about your project proposal.")
    print("Type 'exit' or 'quit' to end the chat.")

    while True:
        prompt = input("\nYour Query > ")
        if prompt.lower() in ["quit", "exit"]:
            print("👋 Session ended. Goodbye!")
            break

        if not prompt.strip():
            continue

        try:
            # RAG process: Retrieve context, then generate answer using LLM
            response = query_engine.query(prompt)

            print("\n🤖 Bot Response:")
            print("-" * 50)
            print(response.response)
            print("-" * 50)

            # Optional: Print the source nodes that were used as context
            # print("\nSource Chunks Used:")
            # for node in response.source_nodes:
            #     print(f"  - Score: {node.get_score():.4f}, Content Snippet: '{node.text[:100]}...'")

        except Exception as e:
            print(f"An error occurred during query generation: {e}")
            print("Ensure Ollama is running and the models are available.")


if __name__ == "__main__":
    # Ensure all models and index are ready before starting the chat
    rag_engine = initialize_rag_system()
    chat_loop(rag_engine)
