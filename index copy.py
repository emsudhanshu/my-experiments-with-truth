import os
import base64
import json
import time
import requests

# --- Configuration ---
# IMPORTANT: When running locally, REPLACE THE EMPTY STRING with your actual API key.
# When running in the Canvas environment, leave this as an empty string ("").
GEMINI_API_KEY = "AIzaSyBzIkcVOKwnK21YP4l5mi79Up_0iV3Wa60" # Local users: Insert your key here.
GEMINI_MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
# Note: The model is gemini-2.5-flash-preview-05-20 for multimodal tasks via the REST API.

def encode_pdf_to_base64(file_path):
    """Reads a PDF file and returns its base64 encoded string."""
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        # Encode the bytes to base64 string
        return base64.b64encode(pdf_bytes).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None

def chat_with_resume(prompt: str, encoded_pdf_data: str, history: list) -> str:
    """
    Sends the user prompt and the resume data to the Gemini API and handles
    exponential backoff for robust communication.
    """
    print("\n[Thinking...]")
    
    # 1. Define System Instruction to guide the model's behavior
    system_instruction = (
        "You are an expert career consultant and personal knowledge base. "
        "Your task is to analyze the provided resume (PDF file) and answer the user's "
        "questions based *only* on the content of the resume. "
        "Maintain a helpful and professional tone. If the information is not in the resume, "
        "politely state that you cannot find it in the document."
        "be natural and human like in your responses."
        "always answer in the simple way possible"
    )

    # 2. Build the current message content
    # The first message in the chat must contain the PDF file.
    if not history:
        # First turn: include both the text prompt and the PDF file
        parts = [
            {"text": prompt},
            {
                "inlineData": {
                    "mimeType": "application/pdf",
                    "data": encoded_pdf_data,
                }
            },
        ]
    else:
        # Subsequent turns: only include the text prompt
        parts = [{"text": prompt}]

    # 3. Construct the full contents list
    contents = history + [{
        "role": "user",
        "parts": parts
    }]

    # 4. Construct the full API payload
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        # Set temperature to 0.1 for factual, concise answers based on the resume
        "generationConfig": {"temperature": 0.1} # Corrected: 'config' changed to 'generationConfig'
    }
    
    # 5. API Call with Exponential Backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Construct the URL
            api_url = GEMINI_MODEL_URL
            if GEMINI_API_KEY:
                api_url = f"{GEMINI_MODEL_URL}?key={GEMINI_API_KEY}"
                
            response = requests.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=30 # Set a timeout for the API call
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Check for candidate and extract text
            if result.get("candidates") and result["candidates"][0].get("content"):
                model_response = result["candidates"][0]["content"]["parts"][0]["text"]
                # Append user and model response to history for context in next turn
                history.append({"role": "user", "parts": parts})
                history.append({"role": "model", "parts": [{"text": model_response}]})
                return model_response
            
            return "The model returned an empty response."

        except requests.exceptions.HTTPError as e:
            # --- CRITICAL DIAGNOSTIC STEP ---
            error_message = f"HTTP Error on attempt {attempt + 1}: {e}"
            
            if response.status_code == 400:
                try:
                    # Print the detailed error message from the server response body
                    detailed_error = response.json().get('error', {}).get('message', 'No detailed message provided.')
                    error_message += f"\n[SERVER DIAGNOSTIC] Code 400 Detail: {detailed_error}"
                except json.JSONDecodeError:
                    error_message += f"\n[SERVER DIAGNOSTIC] Code 400 Detail: Could not decode JSON response body. Status: {response.text}"
            # ---------------------------------
            
            print(error_message)

            if response.status_code == 429 and attempt < max_retries - 1:
                # Handle Rate Limiting (429) with backoff
                sleep_time = 2 ** attempt
                print(f"Rate limit hit. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            elif attempt < max_retries - 1:
                # Handle other 4xx/5xx errors with a small delay
                print(f"Server error. Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return f"Failed to get a response after {max_retries} attempts due to an HTTP error: {e}"
        
        except requests.exceptions.RequestException as e:
            # Handle connection errors, timeouts, etc.
            if attempt < max_retries - 1:
                print(f"Connection Error on attempt {attempt + 1}: {e}. Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return f"Failed to get a response after {max_retries} attempts due to a connection error: {e}"

    return "An unknown error occurred after multiple retries."


def main():
    """Main function to run the interactive chat bot."""
    print("--- Gemini Resume Chatbot ---")
    print("This bot uses the Gemini model to analyze a PDF resume and answer questions about it.")
    print("Type 'exit' or 'quit' to end the session.")
    
    # 1. Hardcode the PDF file path based on user instruction
    pdf_path = "Kakkar,Sudhanshu.pdf"

    # 2. Check if the file exists and its size before proceeding
    if not os.path.exists(pdf_path):
        print(f"\n[FATAL] File not found at '{pdf_path}'. Please ensure the file is in the same directory.")
        return
    
    # Check file size (Warning if > 5MB, as inline data has strict limits for PDF)
    file_size_bytes = os.path.getsize(pdf_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    if file_size_mb > 5:
        print(f"\n[WARNING] The file size is {file_size_mb:.2f} MB. Inline file uploads (like this) are often limited to around 2MB. If you continue to see 400 errors, the file may be too large for the current method.")
    
    # 3. Encode the PDF once
    encoded_pdf_data = encode_pdf_to_base64(pdf_path)
    if not encoded_pdf_data:
        return
    
    # Diagnostic: Print the size of the Base64 string being sent
    encoded_size_kb = len(encoded_pdf_data) / 1024
    print(f"\n[DIAGNOSTIC] Base64 encoded data size: {encoded_size_kb:.2f} KB (Payload size).")

    print(f"\n[SUCCESS] Resume file '{pdf_path}' loaded and encoded successfully.")
    
    # Chat history initialization: keeps track of the conversation
    chat_history = []
    
    # 4. Start the main chat loop
    while True:
        try:
            user_input = input("\n[You] Ask about your resume: ")
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye! Session ended.")
                break
            
            # Use the input and the encoded data to chat
            response = chat_with_resume(user_input, encoded_pdf_data, chat_history)
            print(f"\n[Gemini] {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye! Session ended.")
            break

if __name__ == "__main__":
    main()
