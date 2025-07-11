# frontend/ScholarMate_Streamlit.py

import streamlit as st
import requests
import os
import sys
import random

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
BACKEND_URL = "http://127.0.0.1:8000"

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Constants
LONG_DOC_THRESHOLD = 1000 # characters, a reasonable minimum for LLM to find patterns
NUM_QUESTIONS_PER_TEST = 10 # Display 10 questions per test instance

# --- Session State Initialization for all tabs ---
# Keep track of the text for each tab to avoid re-generating unless PDF changes
if 'last_extracted_text_summary' not in st.session_state:
    st.session_state.last_extracted_text_summary = None
if 'last_extracted_text_glossary' not in st.session_state:
    st.session_state.last_extracted_text_glossary = None
if 'last_extracted_text_qa' not in st.session_state:
    st.session_state.last_extracted_text_qa = None
if 'last_extracted_text_mcq' not in st.session_state:
    st.session_state.last_extracted_text_mcq = None

# For MCQ Test Specifics
if 'all_mcq_questions' not in st.session_state:
    st.session_state.all_mcq_questions = [] # All questions generated from the PDF
if 'current_test_mcqs' not in st.session_state:
    st.session_state.current_test_mcqs = [] # Subset of questions for the current test
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {} # User's answers for current_test_mcqs
if 'test_submitted' not in st.session_state:
    st.session_state.test_submitted = False
if 'shuffled_options_map' not in st.session_state:
    st.session_state.shuffled_options_map = {} # Store shuffled options per question index to keep consistent
if 'test_instance_id' not in st.session_state:
    st.session_state.test_instance_id = 0 # To force a fresh test if needed

# Function to initialize/reset a new test instance
def initialize_new_test_instance(all_questions, num_q_per_test):
    st.session_state.test_instance_id += 1 # Increment to force new test UI state
    st.session_state.test_submitted = False
    st.session_state.user_answers = {}
    st.session_state.shuffled_options_map = {}

    if len(all_questions) > num_q_per_test:
        # Select a random subset of questions
        st.session_state.current_test_mcqs = random.sample(all_questions, num_q_per_test)
    else:
        # If fewer questions than NUM_QUESTIONS_PER_TEST, use all and shuffle
        st.session_state.current_test_mcqs = list(all_questions) # Make a copy
        random.shuffle(st.session_state.current_test_mcqs)

    # Initialize user answers and shuffle options for the new test set
    for i, mcq_item in enumerate(st.session_state.current_test_mcqs):
        st.session_state.user_answers[i] = None # No default selected
        options = mcq_item.get('options', [])
        if options:
            shuffled_opts = list(options) # Make a mutable copy
            random.shuffle(shuffled_opts)
            st.session_state.shuffled_options_map[i] = shuffled_opts
        else:
            st.session_state.shuffled_options_map[i] = []

    st.rerun() # Rerun to display the new test instance

# Function to reset and retake the current test with shuffled order
def retest_current_instance():
    st.session_state.test_submitted = False
    st.session_state.user_answers = {} # Reset user answers
    # Re-shuffle options for the *same* set of current_test_mcqs
    for i, mcq_item in enumerate(st.session_state.current_test_mcqs):
        options = mcq_item.get('options', [])
        if options:
            shuffled_opts = list(options)
            random.shuffle(shuffled_opts)
            st.session_state.shuffled_options_map[i] = shuffled_opts
        else:
            st.session_state.shuffled_options_map[i] = []
    st.rerun()

extracted_text = ""
if uploaded_file is not None:
    st.info("File uploaded successfully! Sending to backend for text extraction...")

    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

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

    if extracted_text:
        st.subheader("Extracted Text Preview:")
        st.text_area("Document Content", extracted_text[:1000], height=200, disabled=True,
                     help=f"Only showing the first 1000 characters. Total characters: {len(extracted_text)}")

        if len(extracted_text) > LONG_DOC_THRESHOLD:
            st.info("Document content is substantial. Proceeding with AI analysis...")

            tabs = st.tabs(["📄 Summary", "📘 Glossary", "❓ Q&A", "🧠 MCQ"])

            # --- Summarization Logic ---
            with tabs[0]: # Summary Tab
                st.subheader("Topic-wise Summary")
                if st.session_state.last_extracted_text_summary != extracted_text or 'summary_text' not in st.session_state:
                    summary_spinner = st.empty()
                    summary_spinner.info("Generating comprehensive summary...")
                    try:
                        summary_response = requests.post(f"{BACKEND_URL}/summarize_document/", json={"text": extracted_text})
                        if summary_response.status_code == 200:
                            summary_data = summary_response.json()
                            st.session_state.summary_text = summary_data.get("summary", "")
                            st.session_state.last_extracted_text_summary = extracted_text
                            summary_spinner.empty()
                            if st.session_state.summary_text:
                                st.markdown(st.session_state.summary_text)
                            else:
                                st.warning("No comprehensive summary could be generated.")
                        else:
                            summary_spinner.error(f"Error generating summary: {summary_response.status_code} - {summary_response.text}")
                    except requests.exceptions.ConnectionError:
                        summary_spinner.error(f"Could not connect to the backend server for summarization at {BACKEND_URL}.")
                    except Exception as e:
                        summary_spinner.error(f"An unexpected error occurred during summarization: {e}")
                elif st.session_state.summary_text:
                    st.markdown(st.session_state.summary_text)
                else:
                    st.warning("No comprehensive summary could be generated yet.")

            # --- Glossary Logic ---
            with tabs[1]: # Glossary Tab
                st.subheader("Technical Glossary (Glassador)")
                
                # Always define the placeholder for the spinner outside the generation logic
                glossary_spinner_placeholder = st.empty()

                # Condition to trigger glossary generation
                # Generate if:
                # 1. extracted_text has changed since last glossary generation (new PDF)
                # 2. 'glossary_content_string' is not yet in session_state (first run/app restart)
                # 3. 'glossary_content_string' is empty but a PDF was uploaded (previous gen failed or returned empty)
                should_generate_glossary = (
                    st.session_state.last_extracted_text_glossary != extracted_text or
                    'glossary_content_string' not in st.session_state or
                    (st.session_state.get('glossary_content_string') == "" and extracted_text)
                )

                if should_generate_glossary:
                    glossary_spinner_placeholder.info("Generating technical glossary...")
                    try:
                        glossary_response = requests.post(f"{BACKEND_URL}/generate_glossary/", json={"text": extracted_text})
                        
                        if glossary_response.status_code == 200:
                            glossary_data = glossary_response.json()
                            # Expecting 'glossary' key to contain the markdown string
                            generated_glossary_str = glossary_data.get("glossary", "") 
                            
                            # Store the string directly
                            st.session_state.glossary_content_string = generated_glossary_str
                            st.session_state.last_extracted_text_glossary = extracted_text
                            glossary_spinner_placeholder.empty() # Clear spinner on success

                            if st.session_state.glossary_content_string:
                                st.success(f"Glossary generated successfully!")
                                # Directly markdown the entire string
                                st.markdown(st.session_state.glossary_content_string) 
                            else:
                                st.warning("No glossary could be generated for this document.")
                        else:
                            glossary_spinner_placeholder.error(f"Error generating glossary: {glossary_response.status_code} - {glossary_response.text}")
                            st.session_state.glossary_content_string = "" # Clear on error
                    except requests.exceptions.ConnectionError:
                        glossary_spinner_placeholder.error(f"Could not connect to the backend server for glossary generation at {BACKEND_URL}.")
                        st.session_state.glossary_content_string = "" # Clear on connection error
                    except Exception as e:
                        glossary_spinner_placeholder.error(f"An unexpected error occurred during glossary generation: {e}")
                        st.session_state.glossary_content_string = "" # Clear on general error
                
                # This block displays the glossary if it's already in session_state
                # and we didn't just try to re-generate it (or re-generation was successful)
                elif st.session_state.get('glossary_content_string'): # Use .get for safety
                    st.markdown(st.session_state.glossary_content_string)
                else: # No glossary content available
                    st.warning("No glossary could be generated yet. Please ensure the document contains technical terms.")

            # # --- Q&A Logic ---
            with tabs[2]: # Q&A Tab
                st.subheader("Self-Testing Questions & Answers")
                if st.session_state.last_extracted_text_qa != extracted_text or 'qa_pairs' not in st.session_state:
                    qa_spinner = st.empty()
                    qa_spinner.info("Generating Q&A pairs... This may take a moment.")
                    try:
                        qa_response = requests.post(f"{BACKEND_URL}/generate_question_and_answer/", json={"text": extracted_text})
                        if qa_response.status_code == 200:
                            qa_data = qa_response.json()
                            st.session_state.qa_pairs = qa_data.get("qa_pairs", [])
                            st.session_state.last_extracted_text_qa = extracted_text
                            qa_spinner.empty()
                            if st.session_state.qa_pairs:
                                st.success(f"Generated {len(st.session_state.qa_pairs)} Q&A pairs!")
                                for i, qa in enumerate(st.session_state.qa_pairs):
                                    st.markdown(f"**Question {i+1}:** {qa.get('question', 'N/A')}")
                                    with st.expander(f"Show Answer for Question {i+1}"):
                                        st.write(qa.get('answer', 'N/A'))
                                    st.markdown("---")
                            else:
                                st.warning("No Q&A pairs could be generated for this document.")
                        else:
                            qa_spinner.error(f"Error generating Q&A: {qa_response.status_code} - {qa_response.text}")
                    except requests.exceptions.ConnectionError:
                        qa_spinner.error(f"Could not connect to the backend server for Q&A generation at {BACKEND_URL}. Please ensure your FastAPI backend is running.")
                    except Exception as e:
                        qa_spinner.error(f"An unexpected error occurred during Q&A generation: {e}")
                elif st.session_state.qa_pairs:
                    for i, qa in enumerate(st.session_state.qa_pairs):
                        st.markdown(f"**Question {i+1}:** {qa.get('question', 'N/A')}")
                        with st.expander(f"Show Answer for Question {i+1}"):
                            st.write(qa.get('answer', 'N/A'))
                        st.markdown("---")
                else:
                    st.warning("No Q&A pairs could be generated yet.")

            # --- MCQ Logic (NEW SECTION for Test) ---
            with tabs[3]: # MCQ Tab
                st.subheader("Multiple Choice Questions (MCQs) - Test Your Knowledge!")

                # Condition to trigger initial MCQ generation from backend
                if st.session_state.last_extracted_text_mcq != extracted_text or not st.session_state.all_mcq_questions:
                    mcq_spinner = st.empty()
                    mcq_spinner.info("Generating ALL possible MCQs from document... This may take a moment.")
                    try:
                        mcq_response = requests.post(
                            f"{BACKEND_URL}/generate_mcq/",
                            json={"text": extracted_text}
                        )
                        if mcq_response.status_code == 200:
                            mcq_data = mcq_response.json()
                            st.session_state.all_mcq_questions = mcq_data.get("mcqs", []) # Store ALL questions
                            st.session_state.last_extracted_text_mcq = extracted_text # Mark as processed
                            mcq_spinner.empty()
                            if st.session_state.all_mcq_questions:
                                st.success(f"Generated {len(st.session_state.all_mcq_questions)} total MCQs! Preparing your test...")
                                # Automatically initialize the first test instance
                                initialize_new_test_instance(st.session_state.all_mcq_questions, NUM_QUESTIONS_PER_TEST)
                            else:
                                st.warning("No MCQs could be generated for this document.")
                        else:
                            mcq_spinner.error(f"Error generating MCQs: {mcq_response.status_code} - {mcq_response.text}")
                    except requests.exceptions.ConnectionError:
                        mcq_spinner.error(f"Could not connect to the backend server for MCQ generation at {BACKEND_URL}.")
                    except Exception as e:
                        mcq_spinner.error(f"An unexpected error occurred during MCQ generation: {e}")
                
                # If we have any MCQs (either newly generated or from session_state)
                if st.session_state.all_mcq_questions:
                    if not st.session_state.current_test_mcqs and not st.session_state.test_submitted:
                        # This handles the case where all_mcq_questions exist, but current_test_mcqs need to be picked
                        initialize_new_test_instance(st.session_state.all_mcq_questions, NUM_QUESTIONS_PER_TEST)
                    
                    if st.session_state.current_test_mcqs:
                        if not st.session_state.test_submitted:
                            st.info(f"Answer the {len(st.session_state.current_test_mcqs)} questions below.")
                            
                            # Display current test questions
                            for i, mcq_item in enumerate(st.session_state.current_test_mcqs):
                                st.markdown(f"**Question {i+1}:** {mcq_item.get('question', 'N/A')}")
                                
                                # Retrieve pre-shuffled options for consistency
                                shuffled_options = st.session_state.shuffled_options_map.get(i, [])

                                # Find current selected index, default to None (no selection) if not found
                                current_selection_index = None
                                if st.session_state.user_answers.get(i) in shuffled_options:
                                    current_selection_index = shuffled_options.index(st.session_state.user_answers.get(i))
                                
                                selected_option = st.radio(
                                    f"Select your answer for Q{i+1} (Test ID: {st.session_state.test_instance_id}):", # Added test ID for debugging
                                    shuffled_options,
                                    index=current_selection_index, # None means no initial selection
                                    key=f"mcq_q_{st.session_state.test_instance_id}_{i}" # Unique key per test instance and question
                                )
                                
                                # Store the user's selected answer
                                st.session_state.user_answers[i] = selected_option
                                st.markdown("---")

                            # Submit Button for the entire test
                            if st.button("Submit Test for Scoring", key="submit_mcq_test"):
                                st.session_state.test_submitted = True
                                st.rerun() # Rerun to display results
                        
                        else: # Test is submitted, display results
                            total_questions = len(st.session_state.current_test_mcqs)
                            correct_count = 0
                            
                            st.subheader("Test Results:")
                            for i, mcq_item in enumerate(st.session_state.current_test_mcqs):
                                user_ans = st.session_state.user_answers.get(i)
                                correct_ans = mcq_item.get('correct_answer', 'N/A')

                                st.markdown(f"**Question {i+1}:** {mcq_item.get('question', 'N/A')}")
                                st.write(f"Your Answer: **{user_ans if user_ans is not None else 'No Answer Selected'}**")
                                st.write(f"Correct Answer: **{correct_ans}**")
                                
                                if str(user_ans).strip().lower() == str(correct_ans).strip().lower():
                                    st.success("🎉 Correct!")
                                    correct_count += 1
                                else:
                                    st.error("❌ Incorrect!")
                                st.markdown("---")
                            
                            st.subheader(f"Final Score: {correct_count} out of {total_questions}")
                            if total_questions > 0:
                                st.progress(correct_count / total_questions)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Retake This Test (Shuffle Order)", key="retest_current"):
                                    retest_current_instance() # Re-shuffles current questions/options
                            with col2:
                                if st.button("Take New Test (Different Questions)", key="new_test"):
                                    initialize_new_test_instance(st.session_state.all_mcq_questions, NUM_QUESTIONS_PER_TEST)

                    else: # No current_test_mcqs selected yet, but all_mcq_questions exist
                        st.warning("No questions are currently loaded for the test. Click below to start.")
                        if st.button(f"Start New Test ({NUM_QUESTIONS_PER_TEST} Questions)", key="start_initial_test"):
                            initialize_new_test_instance(st.session_state.all_mcq_questions, NUM_QUESTIONS_PER_TEST)

                else: # No all_mcq_questions available
                    st.info("Upload a PDF with substantial content to generate MCQs.")
        else:
            st.warning("The document is too short to generate a meaningful summary, glossary, Q&A, or MCQs.")
            st.info("Please upload a PDF with more substantial content (at least 1000 characters) for best results.")

    else:
        st.info("Upload a PDF file above to begin.")

st.sidebar.header("About ScholarMate")
st.sidebar.info(
    "ScholarMate helps you quickly grasp the essence of academic papers "
    "and technical documents by providing concise summaries, a "
    "technical glossary, self-testing Q&A sets, and multiple-choice questions, "
    "all powered by AI."
)
st.sidebar.markdown("---")
st.sidebar.caption("Developed with Streamlit, FastAPI, LangChain, and Groq.")