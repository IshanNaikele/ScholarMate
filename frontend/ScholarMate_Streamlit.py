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

st.set_page_config(page_title="ScholarMate: AI Learning Assistant", layout="wide")
st.title("📚 ScholarMate - Where your Learning becomes easy")
st.markdown("Upload a PDF to get an AI-powered summary and technical glossary.")

# Define the URL of your FastAPI backend server
BACKEND_URL = "http://127.0.0.1:8000" # Ensure this matches your FastAPI server's address

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    st.info("File uploaded successfully! Sending to backend for text extraction...")

    # Prepare the file for sending via requests.post (multipart/form-data)
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

    extracted_text = "" # Initialize extracted_text

    # Use a placeholder for the spinner to update its text
    status_placeholder = st.empty()
    status_placeholder.info("Extracting text from PDF...")

    try:
        response = requests.post(f"{BACKEND_URL}/extract_text/", files=files)

        if response.status_code == 200:
            extracted_data = response.json()
            extracted_text = extracted_data.get("text", "")
            status_placeholder.success("Text extracted successfully!")
        else:
            status_placeholder.error(f"Error extracting text: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        status_placeholder.error(f"Could not connect to the backend server at {BACKEND_URL}. Please ensure your FastAPI backend is running.")
    except Exception as e:
        status_placeholder.error(f"An unexpected error occurred during text extraction: {e}")

    # Display extracted text if available
    if extracted_text:
        st.subheader("Extracted Text Preview:")
        st.text_area("Document Content", extracted_text[:1000], height=200, disabled=True,
                     help=f"Only showing the first 1000 characters. Total characters: {len(extracted_text)}")

        LONG_DOC_THRESHOLD = 1000 # characters, a reasonable minimum for LLM to find patterns

        summary_output = ""
        glossary_output = ""
        qa_pairs_output = [] # Initialize for Q&A output

        if len(extracted_text) > LONG_DOC_THRESHOLD:
            st.info("Document content is substantial. Proceeding with AI analysis...")

            # Create tabs for Summary, Glossary, and Q&A
            tabs = st.tabs(["📄 Summary", "📘 Glossary", "❓ Q&A"])

            # --- Summarization Logic ---
            with tabs[0]: # Summary Tab
                st.subheader("Topic-wise Summary")
                # Removed the button to auto-generate on tab load for better UX
                summary_spinner = st.empty()
                summary_spinner.info("Generating comprehensive summary...")
                try:
                    summary_response = requests.post(
                        f"{BACKEND_URL}/summarize_document/",
                        json={"text": extracted_text}
                    )
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        summary_output = summary_data.get("summary", "")
                        summary_spinner.empty() # Clear spinner on success
                        if summary_output:
                            st.markdown(summary_output)
                        else:
                            st.warning("No comprehensive summary could be generated.")
                    else:
                        summary_spinner.error(f"Error generating summary: {summary_response.status_code} - {summary_response.text}")
                except requests.exceptions.ConnectionError:
                    summary_spinner.error(f"Could not connect to the backend server for summarization at {BACKEND_URL}.")
                except Exception as e:
                    summary_spinner.error(f"An unexpected error occurred during summarization: {e}")

            # --- Glossary Logic ---
            with tabs[1]: # Glossary Tab
                st.subheader("Technical Glossary (Glassador)")
                # Removed the button to auto-generate on tab load for better UX
                glossary_spinner = st.empty()
                glossary_spinner.info("Generating technical glossary...")
                try:
                    glossary_response = requests.post(
                        f"{BACKEND_URL}/generate_glossary/",
                        json={"text": extracted_text}
                    )
                    if glossary_response.status_code == 200:
                        glossary_data = glossary_response.json()
                        glossary_output = glossary_data.get("glossary", "")
                        glossary_spinner.empty() # Clear spinner on success
                        if glossary_output:
                            st.markdown(glossary_output)
                        else:
                            st.warning("No glossary available.")
                    else:
                        glossary_spinner.error(f"Error generating glossary: {glossary_response.status_code} - {glossary_response.text}")
                except requests.exceptions.ConnectionError:
                    glossary_spinner.error(f"Could not connect to the backend server for glossary generation at {BACKEND_URL}.")
                except Exception as e:
                    glossary_spinner.error(f"An unexpected error occurred during glossary generation: {e}")

            # --- Q&A Logic ---
            with tabs[2]: # Q&A Tab
                st.subheader("Self-Testing Questions & Answers")
                if st.button("Generate Q&A Pairs", key="generate_qa_button"):
                    qa_spinner = st.empty()
                    qa_spinner.info("Generating Q&A pairs... This may take a moment.")
                    try:
                        qa_response = requests.post(
                            f"{BACKEND_URL}/generate_question_and_answer/", # The new endpoint
                            json={"text": extracted_text}
                        )
                        if qa_response.status_code == 200:
                            qa_data = qa_response.json()
                            qa_pairs_output = qa_data.get("qa_pairs", []) # Expecting a list of dicts
                            qa_spinner.empty() # Clear spinner on success

                            if qa_pairs_output:
                                st.success(f"Generated {len(qa_pairs_output)} Q&A pairs!")
                                for i, qa in enumerate(qa_pairs_output):
                                    st.markdown(f"**Question {i+1}:** {qa.get('question', 'N/A')}")
                                    # Use st.expander to hide/show the answer
                                    with st.expander(f"Show Answer for Question {i+1}"):
                                        st.write(qa.get('answer', 'N/A'))
                                    st.markdown("---") # Visual separator
                            else:
                                st.warning("No Q&A pairs could be generated for this document.")
                        else:
                            qa_spinner.error(f"Error generating Q&A: {qa_response.status_code} - {qa_response.text}")
                    except requests.exceptions.ConnectionError:
                        qa_spinner.error(f"Could not connect to the backend server for Q&A generation at {BACKEND_URL}. Please ensure your FastAPI backend is running.")
                    except Exception as e:
                        qa_spinner.error(f"An unexpected error occurred during Q&A generation: {e}")

        else:
            st.warning("The document is too short to generate a meaningful summary, glossary, or Q&A.")
            st.info("Please upload a PDF with more substantial content (at least 1000 characters) for best results.")

    else:
        st.info("Upload a PDF file above to begin.")

st.sidebar.header("About ScholarMate")
st.sidebar.info(
    "ScholarMate helps you quickly grasp the essence of academic papers "
    "and technical documents by providing concise summaries, a "
    "technical glossary, and self-testing Q&A sets, all powered by AI."
)
st.sidebar.markdown("---")
st.sidebar.caption("Developed with Streamlit, FastAPI, LangChain, and Groq.")