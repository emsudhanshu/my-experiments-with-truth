import streamlit as st
import os
import base64
import json
import time
import requests
from typing import List, Dict, Any

# --- Configuration (Hardcoded as requested) ---
# NOTE: Using the key provided in your last message. Please ensure it is a valid, live key.
# WARNING: Storing keys directly in code is unsafe for production environments.
GEMINI_API_KEY = "AIzaSyBzIkcVOKwnK21YP4l5mi79Up_0iV3Wa60"
# NOTE: Set your resume file path here.
RESUME_FILE_PATH = "./Kakkar,Sudhanshu.pdf" 

# Note: The model is gemini-2.5-flash-preview-05-20 for multimodal tasks via the REST API.
GEMINI_MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Helper Functions ---

def encode_local_file_to_base64(file_path):
    """Reads a local file and returns its base64 encoded string."""
    try:
        # Check if file exists and provide diagnostic path
        absolute_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            # This error is now handled in the main setup block for better UI flow
            return None
            
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
            
        # Encode the bytes to base64 string
        return base64.b64encode(pdf_bytes).decode("utf-8")
        
    except Exception as e:
        st.error(f"An error occurred while reading or encoding the PDF: {e}")
        return None

def generate_response(prompt: str, encoded_pdf_data: str, history: List[Dict[str, Any]], api_key: str) -> str:
    """
    Sends the prompt, history, and resume data to the Gemini API.
    Handles conversation turn context and exponential backoff.
    """
    # The key check is now handled in the setup block, but we keep a final runtime check
    if not api_key:
        return "API Key is missing. Please ensure it is correctly hardcoded."

    # 1. Define System Instruction to guide the model's behavior (Refined for stricter factual output)
    system_instruction = (
        "You are a strict, expert career consultant who operates as a knowledge base. "
        "Your responses **MUST** be factual and derived **EXCLUSIVELY** from the provided resume (PDF file). "
        "Do not use external knowledge. For any detail not explicitly found in the document, "
        "you must clearly and politely state that the information is unavailable in the resume."
    )

    # 2. Build the current message content
    current_user_message_parts = [{"text": prompt}]

    # Check if this is the first turn (history only contains system instructions or is empty)
    is_first_turn = True
    for msg in history:
        if msg.get("role") == "user":
            is_first_turn = False
            break

    # Only attach the PDF data to the very first user message
    if is_first_turn and encoded_pdf_data:
        current_user_message_parts.append(
            {
                "inlineData": {
                    "mimeType": "application/pdf",
                    "data": encoded_pdf_data,
                }
            }
        )

    # 3. Prepare contents for the API call
    api_history = []
    for message in history:
        # Map Streamlit's 'assistant' role to the Gemini API's 'model' role
        role = message["role"]
        api_role = "model" if role == "assistant" else "user"
        
        # Only include the text part for subsequent history messages
        text_part = message.get("content", "")
        if text_part and isinstance(text_part, str):
            api_history.append({"role": api_role, "parts": [{"text": text_part}]})

    # Add the current user message
    contents = api_history + [{
        "role": "user",
        "parts": current_user_message_parts
    }]

    # 4. Construct the full API payload
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {"temperature": 0.1}
    }

    # 5. API Call with Exponential Backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            api_url = f"{GEMINI_MODEL_URL}?key={api_key}"

            response = requests.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=45 
            )
            response.raise_for_status()

            result = response.json()

            if result.get("candidates") and result["candidates"][0].get("content"):
                return result["candidates"][0]["content"]["parts"][0]["text"]

            return "The model returned an empty response. Check the console for errors."

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
            elif attempt < max_retries - 1 and response.status_code >= 500:
                time.sleep(2) 
            else:
                try:
                    error_detail = response.json().get('error', {}).get('message', 'No detail available.')
                    return f"API Error ({response.status_code}): {error_detail}"
                except:
                    return f"API Error ({response.status_code}): Could not parse error message."

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return f"Connection Error: Failed after {max_retries} attempts."

    return "An unknown error occurred."


# --- Streamlit App Layout ---

st.set_page_config(page_title="Gemini Resume Analyzer", layout="wide")

st.title("📄 AI Resume Chatbot")
st.markdown("This chatbot uses a hardcoded PDF file and API key for a streamlined experience.")
st.divider()

# --- Initial Setup (Hardcoded File and Key Handling) ---

# Check if file has been processed in session state
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
    st.session_state.messages = []
    st.session_state.encoded_pdf = None
    st.session_state.setup_status_ran = False

# Process file only once upon the first run
if not st.session_state.file_processed:
    with st.sidebar:
        st.header("Setup Status")
        st.info("Attempting to load hardcoded file and configuration...")
        
        # 1. API Key Check
        if GEMINI_API_KEY == "YOUR_HARDCODED_GEMINI_API_KEY_HERE":
            st.error("Setup Failed: Please replace the placeholder API key with your actual key.")
            st.session_state.setup_status_ran = True
        
        # 2. File Path Check
        elif not os.path.exists(RESUME_FILE_PATH):
            st.error("Setup Failed: File Not Found!")
            st.warning(f"The application is looking for **`{RESUME_FILE_PATH}`** at this location: **`{os.path.abspath(RESUME_FILE_PATH)}`**.")
            st.error("Please move the PDF file into the same directory as this script.")
            st.session_state.setup_status_ran = True

        # 3. Successful Load Attempt
        else:
            st.session_state.encoded_pdf = encode_local_file_to_base64(RESUME_FILE_PATH)
            
            if st.session_state.encoded_pdf:
                st.session_state.file_processed = True
                st.success(f"File '{RESUME_FILE_PATH}' loaded and encoded ({len(st.session_state.encoded_pdf)//1024} KB).")
                st.success(f"API Key set internally. App is ready to chat!")
            else:
                st.error("Setup failed. Failed to encode PDF. Check the console for details.")
            st.session_state.setup_status_ran = True


# 1. Sidebar for Configuration (Display status only)
with st.sidebar:
    if st.session_state.file_processed:
        st.header("Configuration")
        st.code(f"API Key: Set (Internal)")
        st.code(f"Resume: {RESUME_FILE_PATH}")
    elif st.session_state.setup_status_ran:
        st.warning("Setup needs attention. See errors above.")


# 2. Main Chat Interface

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
is_disabled = not st.session_state.file_processed or GEMINI_API_KEY == "AIzaSyBzIkcVOKwnK21YP4l5mi79Up_0iV3Wa60"

if prompt := st.chat_input("Ask a question about the resume...", disabled=is_disabled):
    
    if not st.session_state.file_processed:
        st.error("Cannot chat. File failed to load during setup.")
    else:
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the model
        with st.chat_message("assistant"):
            with st.spinner("Analyzing resume..."):
                response = generate_response(
                    prompt, 
                    st.session_state.encoded_pdf, 
                    st.session_state.messages, 
                    GEMINI_API_KEY # Use the hardcoded key
                )
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
