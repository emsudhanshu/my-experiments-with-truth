import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Changed from langchain_openai to langchain_community for Ollama
from langchain_community.embeddings import OllamaEmbeddings # <--- OLLAMA IMPORT
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- CONFIGURATION ---
FILE_PATH = "./Kakkar,Sudhanshu.pdf"
CHROMA_DB_PATH = "./chroma_db_resume"
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "Ollama default (usually llama2)" 

def setup_rag_chain():
    """Sets up the LLM, RAG components, and the final LangChain retrieval chain."""
    
    # ----------------------------------------------------------------------
    # FIX 1: Read the key and check for both common variable names.
    # ----------------------------------------------------------------------
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # --- API Key Check ---
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not found.")
        print("Please set your Google Gemini API key before running the script.")
        print("NOTE: OllamaEmbeddings also requires the Ollama server to be running locally.")
        sys.exit(1)

    print("1. Initializing LLM and Embeddings...")
    
    # ----------------------------------------------------------------------
    # FIX 2: Explicitly pass the API key to the LLM to resolve the 403 error.
    # ----------------------------------------------------------------------
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=gemini_key  # <-- CRUCIAL: Ensures correct authentication scope
    )
    
    # Initialize OllamaEmbeddings
    # Ollama server MUST be running for this to work.
    try:
        embeddings = OllamaEmbeddings()
    except Exception as e:
        print("ERROR: Could not initialize OllamaEmbeddings.")
        print("Ensure the Ollama server is running locally and an embedding model is pulled (e.g., 'ollama pull llama2').")
        print(f"Details: {e}")
        sys.exit(1)

    # --- Data Ingestion and Indexing ---
    print(f"2. Loading document: {FILE_PATH}...")
    try:
        loader = PyPDFLoader(FILE_PATH)
        documents = loader.load()
    except Exception as e:
        print(f"ERROR: Could not load the PDF file. Make sure '{FILE_PATH}' exists.")
        print(f"Details: {e}")
        sys.exit(1)

    print("3. Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)

    print(f"Created {len(docs)} document chunks.") # Added a confirmation print

    print("4. Creating/Loading Vector Store (ChromaDB)...")
    # This automatically creates the vectors if the directory is new
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    # --- LangChain Chain Setup ---
    retriever = vectorstore.as_retriever()
    
    system_prompt = (
        "You are an expert Resume Analyst. "
        "Answer the user's question only based on the provided resume context. "
        "If the answer is not in the context, state that you cannot find the information in the resume. "
        "Do not use external knowledge. "
        "\n\nCONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

def run_qa_bot(rag_chain):
    """Runs the interactive question and answer loop."""
    print("\n" + "="*50)
    print("      🚀 Personalized Resume Q&A Bot is Ready 🚀")
    print("="*50)
    print("Ask questions about the resume. Type 'exit' or 'quit' to stop.")
    print("NOTE: Ensure Ollama is running and accessible (e.g., via http://localhost:11434).")

    while True:
        question = input("\nYour Question: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye! The ChromaDB vector store is saved locally.")
            break
        
        try:
            # Invoke the RAG chain with the user's question
            print("Processing...")
            response = rag_chain.invoke({"input": question})

            answer = response["answer"]
            
            # Extract unique sources for reference
            sources = [doc.metadata.get('source', 'Unknown Source') for doc in response["context"]]
            unique_sources = list(set(sources))
            
            print("\n[BOT] Answer:", answer)
            print("\n[BOT] Source Files Used:", unique_sources)

        except Exception as e:
            print(f"\n[BOT] An error occurred: {e}")

if __name__ == "__main__":
    rag_chain = setup_rag_chain()
    run_qa_bot(rag_chain)