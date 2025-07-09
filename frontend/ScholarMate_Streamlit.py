# frontend/ScholarMate_Streamlit.py

import streamlit as st
import requests
import os
import sys

# --- CRUCIAL FIX for ModuleNotFoundError (Keep this) ---
# Ensures the project root is in sys.path so internal modules can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END CRUCIAL FIX ---

st.title("Welcome to ScholarMate - Where your Learning becomes easy ")

# Define the URL of your FastAPI backend server
BACKEND_URL = "http://127.0.0.1:8000" # Ensure this matches your FastAPI server's address

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    st.write("File uploaded successfully! Sending to backend for text extraction...")

    # Prepare the file for sending via requests.post (multipart/form-data)
    # The key "file" must match the parameter name in your FastAPI endpoint (file: UploadFile)
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

    extracted_text = "" # Initialize extracted_text to handle cases where extraction fails
    with st.spinner("Extracting text..."):
        try:
            # Make a POST request to your FastAPI backend's /extract_text/ endpoint
            response = requests.post(f"{BACKEND_URL}/extract_text/", files=files)

            if response.status_code == 200:
                # If extraction was successful, parse the JSON response
                extracted_data = response.json()
                extracted_text = extracted_data.get("text", "")
            else:
                # Handle errors from the backend's text extraction endpoint
                st.error(f"Error extracting text: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            # Handle cases where the Streamlit app cannot connect to the backend
            st.error(f"Could not connect to the backend server at {BACKEND_URL}. Please ensure your FastAPI backend is running.")
        except Exception as e:
            # Catch any other unexpected errors during the request
            st.error(f"An unexpected error occurred during text extraction: {e}")

    # Display extracted text if available
    if extracted_text:
        st.subheader("Extracted Text (First 1000 Chars):")
        st.text_area("Document Content", extracted_text[:1000], height=300)
        if len(extracted_text) > 1000:
            st.info(f"Only showing the first 1000 characters. Total characters: {len(extracted_text)}")

        # --- Summarization Logic ---
        # Define a threshold for what constitutes a "long" document for summarization
        # This is arbitrary; adjust based on your needs and expected document lengths.
        # 5000 characters is a rough estimate for when chunking might be beneficial.
        LONG_DOC_THRESHOLD = 5000 # characters

        if len(extracted_text) > LONG_DOC_THRESHOLD:
            st.write("Document is long. Sending for comprehensive summarization...")
            with st.spinner("Generating comprehensive summary (this may take a while for large documents)..."):
                try:
                    # Send the full extracted text to the new summarization endpoint
                    # This is sent as a JSON body, not multipart/form-data, as it's just text
                    summary_response = requests.post(
                        f"{BACKEND_URL}/summarize_document/",
                        json={"text": extracted_text} # Send the text in a JSON payload
                    )

                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        final_summary = summary_data.get("summary", "")
                        if final_summary:
                            st.subheader("Comprehensive Summary:")
                            st.markdown(final_summary) # Use markdown for better formatting
                        else:
                            st.warning("No comprehensive summary could be generated.")
                    else:
                        st.error(f"Error generating summary: {summary_response.status_code} - {summary_response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to the backend server for summarization at {BACKEND_URL}.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during summarization: {e}")
        else:
            # For shorter documents, you might call a different endpoint or use a simpler chain
            st.write("Document is relatively short. You could process it directly with a simpler summarization model if desired.")
            # Example: If you wanted to use the 'pdf_summary_chain_short' from app.py
            # try:
            #     short_summary_response = requests.post(
            #         f"{BACKEND_URL}/pdf-summary-short/invoke",
            #         json={"input": {"input_text": extracted_text}}
            #     )
            #     if short_summary_response.status_code == 200:
            #         st.subheader("Short Summary:")
            #         st.markdown(short_summary_response.json().get("output"))
            #     else:
            #         st.error(f"Error short summary: {short_summary_response.status_code}")
            # except:
            #     st.info("Skipped short summarization example for now.")
            st.info("No short summarization applied for this example.")