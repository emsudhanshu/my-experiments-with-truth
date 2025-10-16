import os
import sys
import shutil 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# --- CONFIGURATION ---
# NOTE: The file name is updated here to match the attached document's name
FILE_PATH = "project_document.pdf" 
CHROMA_DB_PATH = "./chroma_db_resume"
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "Ollama default (usually llama2)"

# Global variable to store chat history
chat_history: list[BaseMessage] = []

def setup_conversational_rag_chain():
    """Sets up the LLM, RAG components, and the final LangChain conversational retrieval chain."""
    
    # --- API Key Check ---
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not found.")
        print("Please set your Google Gemini API key before running the script.")
        sys.exit(1)

    print("1. Initializing LLM and Embeddings...")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=gemini_key
    )
    
    # Initialize OllamaEmbeddings
    try:
        # Note: You must have Ollama server running and a model pulled (e.g., 'ollama pull llama2')
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
        # NOTE: For local execution, you MUST ensure the specified PDF file exists in the directory.
        print(f"ERROR: Could not load the PDF file. Make sure '{FILE_PATH}' exists.")
        print(f"Details: {e}")
        sys.exit(1)

    # --- FINE TUNED CHUNKING (Optimized for concise facts) ---
    print("3. Splitting text into chunks with fine tuned settings (chunk_size=400)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, # Reduced for fact precision
        chunk_overlap=80, 
        length_function=len
    )
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} document chunks.")

    # 4. Creating/Loading Vector Store (ChromaDB)
    print("4. Creating NEW Vector Store (ChromaDB) with fresh index...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    # *** CRITICAL FIX: Increased retrieval (k=3) ***
    # This grabs 3 chunks to ensure full lists (like team members) are retrieved.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    
    # --- LangChain Chain Setup for CONVERSATION ---
    
    # 1. History Aware Retriever Chain: Condenses question and history into a standalone search query.
    history_aware_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a conversational history and a follow up question, rephrase the follow up question "
                "to be a standalone question for document retrieval. Do NOT answer the question, just rephrase it. "
                "Example: 'Where did he work next?' -> 'What was his next job after [previous job]?'.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, history_aware_prompt
    )

    # 2. Document Chain: Answers the question using the retrieved context.
    # --- REVISED SYSTEM PROMPT ---
    document_chain_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Master's Project Analyst. "
                "Your task is to answer the user's questions about the project proposal using **only** the provided document context. "
                "If the information is not in the context, state clearly that you cannot find the details in the project proposal. "
                "Do not use any external knowledge. "
                "Ensure all responses adhere to the user's custom instruction: **Never use hyphens when responding**. "
                "For simple, factual questions (like 'What are the team members names?' or 'What is the deadline for data cleaning?'), provide the answer **directly, concisely, and without introductory phrases or extra sentences**. "
                "Example: 'Fiza Pathan, Azizul Haque, Sudhanshu Kakkar' or '03/31/2025'."
                "\n\nCONTEXT:\n{context}",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, document_chain_prompt)

    # 3. Conversational Retrieval Chain: Combines the history aware retriever and the document chain.
    conversational_rag_chain = create_retrieval_chain(
        history_aware_retriever, document_chain
    )
    
    return conversational_rag_chain

def run_qa_bot(rag_chain):
    """Runs the interactive question and answer loop, managing chat history."""
    global chat_history
    print("\n" + "="*50)
    print("      🚀 Conversational Project Q&A Bot is Ready 🚀")
    print("="*50)
    print("Ask questions about the project document. Followup questions are now supported! Type 'exit' or 'quit' to stop.")
    print("NOTE: Ensure Ollama is running and accessible (e.g., via http://localhost:11434).")

    while True:
        question = input("\nYour Question: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye! The ChromaDB vector store is saved locally.")
            break
        
        try:
            # Invoke the RAG chain with the user's question AND chat history
            print("Processing...")
            response = rag_chain.invoke(
                {"input": question, "chat_history": chat_history}
            )

            answer = response["answer"]
            
            # CRUCIAL STEP: Update chat history with the new turn for memory
            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=answer))

            # Extract unique sources for reference
            sources = [doc.metadata.get('source', 'Unknown Source') for doc in response.get("context", [])]
            unique_sources = list(set(sources))
            
            print("\n[BOT] Answer:", answer)
            print("\n[BOT] Source Files Used:", unique_sources)

        except Exception as e:
            print(f"\n[BOT] An error occurred: {e}")

if __name__ == "__main__":
    # CRITICAL FIX IMPLEMENTATION
    # Automatically delete the old vector store to guarantee a fresh index with k=3
    if os.path.exists(CHROMA_DB_PATH):
        print(f"4a. Found existing database. Deleting '{CHROMA_DB_PATH}' to ensure fresh index.")
        try:
            shutil.rmtree(CHROMA_DB_PATH)
        except OSError as e:
            print(f"Error: Could not delete old database directory: {e.strerror}. Cannot proceed.")
            sys.exit(1)

    rag_chain = setup_conversational_rag_chain()
    run_qa_bot(rag_chain)