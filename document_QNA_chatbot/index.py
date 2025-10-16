# This script implements a simple Retrieval-Augmented Generation (RAG) system
# with conversation memory for a Proof of Concept (PoC) using LlamaIndex and Ollama.
#
# PREREQUISITES:
# 1. Ollama server must be running locally (usually on port 11434).
# 2. The required models must be pulled:
#    ollama pull llama3
#    ollama pull nomic-embed-text
# 3. Install Python dependencies:
#    pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama pydantic
#    pip install pypdf llama-index-readers-file
#
# NOTE: This version replaces the simple QueryEngine with a ChatEngine to enable follow up questions.

import os
import sys
# Import necessary LlamaIndex components for chat and indexing
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# We will create the chat engine directly from the index, making the CondenseQuestionChatEngine import optional
from llama_index.llms.ollama import Ollama as OllamaLLM
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Configuration ---
# Ollama runs on http://localhost:11434 by default.
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3" # LLM for generation
EMBEDDING_MODEL = "nomic-embed-text" # Embedding model for vector creation

# Path to the PDF file to be loaded
PDF_FILE_PATH = "project_document.pdf" 

def initialize_rag_system():
    """Initializes the Ollama models and creates the VectorStoreIndex."""
    print("--- 1. Initializing RAG System ---")

    # 1. Setup Ollama LLM and Embedding Model
    try:
        # Use a single instance for both RAG and chat context condensation
        llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        print(f"✅ Ollama LLM ({LLM_MODEL}) and Embedding ({EMBEDDING_MODEL}) configured.")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama. Is the server running at {OLLAMA_BASE_URL}? (Error: {e})")
        if "model" in str(e) and "not found" in str(e):
             print(f"HINT: Please ensure you ran 'ollama pull {LLM_MODEL}' and 'ollama pull {EMBEDDING_MODEL}' in your terminal.")
        sys.exit(1)

    # 2. Load the Document from PDF
    print(f"--- 2. Loading Document from {PDF_FILE_PATH} ---")
    try:
        documents = SimpleDirectoryReader(input_files=[PDF_FILE_PATH]).load_data()
        print(f"✅ Document loaded successfully. Found {len(documents)} text chunks/pages.")
    except Exception as e:
        print(f"❌ Failed to load document from {PDF_FILE_PATH}. (Error: {e})")
        print("HINT: Ensure the 'project_document.pdf' file exists in the same directory as the script, and you have installed 'pypdf' and 'llama-index-readers-file'.")
        sys.exit(1)


    # 3. Create Index and Vector Database
    print("--- 3. Creating Index and Vector Database ---")
    try:
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            show_progress=True
        )
        print("✅ Index created successfully. Document embeddings stored.")
    except Exception as e:
        print(f"❌ Failed to create the index. (Error: {e})")
        sys.exit(1)

    # 4. Create the Chat Engine (RAG pipeline with memory)
    # Use index.as_chat_engine() which automatically uses the CondensePlusContext mode 
    # for follow-up questions, simplifying the setup and avoiding the TypeError.
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question", # This mode provides conversation memory
        llm=llm,
        similarity_top_k=3, # Retriever setting
        verbose=True # Set to True to see the rewritten question
    )
    
    print("✅ Chat Engine with Conversation Memory is ready.")

    return chat_engine

def chat_loop(chat_engine):
    """Starts the interactive chat loop."""
    print("\n" * 2)
    print("==============================================")
    print(f"🤖 Conversational RAG Bot (Ollama/{LLM_MODEL})")
    print("==============================================")
    print("Ask me anything, including follow up questions!")
    print("Type 'exit' or 'quit' to end the chat.")

    while True:
        prompt = input("\nYour Query > ")
        if prompt.lower() in ["quit", "exit"]:
            print("👋 Session ended. Goodbye!")
            break

        if not prompt.strip():
            continue

        try:
            # The chat engine manages the conversation history automatically
            response = chat_engine.chat(prompt)

            print("\n🤖 Bot Response:")
            print("-" * 50)
            print(response.response)
            print("-" * 50)

            # NOTE: If verbose=True is set on the chat_engine, you will see the
            # rewritten question (the 'context condensation') in the console output.

        except Exception as e:
            print(f"An error occurred during query generation: {e}")
            print("Ensure Ollama is running and the models are available.")


if __name__ == "__main__":
    # Ensure all models and index are ready before starting the chat
    rag_engine = initialize_rag_system()
    chat_loop(rag_engine)
