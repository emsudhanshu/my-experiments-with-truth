import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- CONFIGURATION ---
FILE_PATH = "./Kakkar,Sudhanshu.pdf"
CHROMA_DB_PATH = "./chroma_db_resume"
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"

def setup_rag_chain():
    """Sets up the LLM, RAG components, and the final LangChain retrieval chain."""
    
    # Check for API Key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not found.")
        print("Please set your API key before running the script.")
        sys.exit(1)

    print("1. Initializing LLM and Embeddings...")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
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
    print("      🚀 Personalized Resume Q&A Bot is Ready 🚀")
    print("="*50)
    print("Ask questions about the resume. Type 'exit' or 'quit' to stop.")

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