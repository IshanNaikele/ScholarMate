
# frontend/ScholarMate_Streamlit.py

import streamlit as st
import requests
import os
import sys
import random
import hashlib

# --- CRUCIAL FIX for ModuleNotFoundError (Fixed __file__) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END CRUCIAL FIX ---

st.set_page_config(page_title="ScholarMate: AI Learning Assistant", layout="wide")
st.title("📚 ScholarMate - Where your Learning becomes easy")
# Updated markdown to include the new text input option
st.markdown("Upload a PDF, provide a YouTube URL, or paste text directly to get AI-powered insights.")

from dotenv import load_dotenv # <<< Make sure this is imported
load_dotenv()

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.00.1:8000") 

# --- Inputs ---
# Added a new text_area for direct user input
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
youtube_url_input = st.text_input("Or enter a YouTube video URL", placeholder="e.g., https://www.youtube.com/watch?v=EjavYOFOJJo")
st.markdown("<h5 style='text-align: center; color: grey;'>OR</h5>", unsafe_allow_html=True)
text_input = st.text_area("Paste your text here", placeholder="You can paste any text content here to get started.", height=150)
st.markdown("---")


# Session state setup
def init_session():
    # Initialize all necessary session state variables
    if 'last_processed_content_hash' not in st.session_state:
        st.session_state.last_processed_content_hash = None
    if 'full_text' not in st.session_state: # This will store the actual content
        st.session_state.full_text = ""
    if 'content_source' not in st.session_state: # To track if it's PDF, YouTube, or Text
        st.session_state.content_source = None

    # Tab-specific content caches
    if 'summary_text' not in st.session_state:
        st.session_state.summary_text = None
    if 'glossary_content_string' not in st.session_state:
        st.session_state.glossary_content_string = None
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = None

    # MCQ Test Specifics
    if 'all_mcq_questions' not in st.session_state:
        st.session_state.all_mcq_questions = []
    if 'current_test_mcqs' not in st.session_state:
        st.session_state.current_test_mcqs = []
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'test_submitted' not in st.session_state:
        st.session_state.test_submitted = False
    if 'shuffled_options_map' not in st.session_state:
        st.session_state.shuffled_options_map = {}
    if 'test_instance_id' not in st.session_state:
        st.session_state.test_instance_id = 0

init_session()

LONG_DOC_THRESHOLD = 1000
NUM_QUESTIONS_PER_TEST = 10

# Initialize a new MCQ test
def initialize_new_test_instance(all_questions, num_q_per_test):
    st.session_state.test_instance_id += 1
    st.session_state.test_submitted = False
    st.session_state.user_answers = {}
    st.session_state.shuffled_options_map = {}

    if not all_questions:
        st.warning("No questions available to create a test.")
        st.session_state.current_test_mcqs = []
        return

    if len(all_questions) > num_q_per_test:
        st.session_state.current_test_mcqs = random.sample(all_questions, num_q_per_test)
    else:
        st.session_state.current_test_mcqs = list(all_questions)
        random.shuffle(st.session_state.current_test_mcqs)

    for i, mcq_item in enumerate(st.session_state.current_test_mcqs):
        st.session_state.user_answers[i] = None
        options = mcq_item.get('options', [])
        if options:
            shuffled_opts = list(options)
            random.shuffle(shuffled_opts)
            st.session_state.shuffled_options_map[i] = shuffled_opts
        else:
            st.session_state.shuffled_options_map[i] = []

    st.rerun()


def retest_current_instance():
    st.session_state.test_submitted = False
    st.session_state.user_answers = {}
    for i, mcq_item in enumerate(st.session_state.current_test_mcqs):
        options = mcq_item.get('options', [])
        if options:
            shuffled_opts = list(options)
            random.shuffle(shuffled_opts)
            st.session_state.shuffled_options_map[i] = shuffled_opts
        else:
            st.session_state.shuffled_options_map[i] = []
    st.rerun()

# --- Content Fetching Logic ---
# Determine the current input hash based on the active input field
current_input_hash = None
# Added a condition to check the new text_input field
if uploaded_file is not None:
    # For PDF, hash the file content for robust change detection
    current_input_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
elif youtube_url_input:
    # For YouTube, hash the URL string
    current_input_hash = hashlib.md5(youtube_url_input.encode('utf-8')).hexdigest()
elif text_input:
    # For direct text input, hash the text content
    current_input_hash = hashlib.md5(text_input.encode('utf-8')).hexdigest()


# Check if content input has changed
if current_input_hash != st.session_state.last_processed_content_hash:
    # Reset all relevant session states if input changes
    st.session_state.last_processed_content_hash = current_input_hash
    st.session_state.full_text = "" # Clear cached text
    st.session_state.content_source = None # Clear cached source
    st.session_state.summary_text = None
    st.session_state.glossary_content_string = None
    st.session_state.qa_pairs = None
    st.session_state.all_mcq_questions = []
    st.session_state.current_test_mcqs = []
    st.session_state.user_answers = {}
    st.session_state.test_submitted = False
    st.session_state.shuffled_options_map = {}
    st.session_state.test_instance_id = 0

    # Process new input
    if uploaded_file is not None:
        st.info("PDF file detected. Extracting text...")
        status_placeholder = st.empty()
        status_placeholder.info("Extracting text from PDF...")
        try:
            response = requests.post(f"{BACKEND_URL}/extract_text/", files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")})
            if response.status_code == 200:
                st.session_state.full_text = response.json().get("text", "")
                st.session_state.content_source = 'pdf'
                status_placeholder.success("Text extracted from PDF successfully!")
            else:
                status_placeholder.error(f"Failed to extract PDF: {response.status_code} - {response.text}")
                st.session_state.full_text = ""
                st.session_state.content_source = None
        except requests.exceptions.ConnectionError:
            status_placeholder.error(f"Could not connect to the backend server at {BACKEND_URL}. Please ensure your FastAPI backend is running.")
            st.session_state.full_text = ""
            st.session_state.content_source = None
        except Exception as e:
            status_placeholder.error(f"An unexpected error occurred during PDF text extraction: {e}")
            st.session_state.full_text = ""
            st.session_state.content_source = None

    elif youtube_url_input:
        st.info("YouTube URL detected. Fetching transcript...")
        status_placeholder = st.empty()
        status_placeholder.info("Fetching YouTube transcript...")
        try:
            response = requests.post(f"{BACKEND_URL}/get_youtube_transcript/", json={"youtube_url": youtube_url_input}, timeout=180)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            st.session_state.full_text = response.json().get("transcript", "")
            st.session_state.content_source = 'youtube'
            status_placeholder.success("YouTube transcript fetched successfully!")
        except requests.exceptions.HTTPError as e:
            error_detail = "An unknown error occurred."
            try:
                error_json = e.response.json()
                error_detail = error_json.get("detail", f"Backend error: {e.response.status_code}")
            except ValueError:
                error_detail = e.response.text
            status_placeholder.error(f"Error fetching YouTube transcript: {e.response.status_code} - {error_detail}")
            st.session_state.full_text = ""
            st.session_state.content_source = None
        except requests.exceptions.ConnectionError:
            status_placeholder.error(f"Could not connect to the backend server at {BACKEND_URL}. Please ensure your FastAPI backend is running.")
            st.session_state.full_text = ""
            st.session_state.content_source = None
        except requests.exceptions.Timeout:
            status_placeholder.error("Request timed out while fetching YouTube transcript. The video might be too long or the network is slow.")
            st.session_state.full_text = ""
            st.session_state.content_source = None
        except Exception as e:
            status_placeholder.error(f"An unexpected error occurred during YouTube transcript fetching: {e}")
            st.session_state.full_text = ""
            st.session_state.content_source = None
    
    # Added a new block to handle the direct text input
    elif text_input:
        st.info("Text input detected. Processing content...")
        st.session_state.full_text = text_input
        st.session_state.content_source = 'text' # Set the source to 'text'
        st.success("Text content loaded successfully!")

# Use the full_text from session state for display and further processing
full_text = st.session_state.full_text
content_source = st.session_state.content_source

# --- Display Content and Tabs (Only if full_text is available) ---
if full_text and content_source:
    # Updated subheader to dynamically show the content source
    st.subheader(f"Extracted Content Preview ({content_source.upper()}):")
    st.text_area("Content", full_text[:1000], height=200, disabled=True,
                 help=f"Only showing the first 1000 characters. Total characters: {len(full_text)}")

    if len(full_text) > LONG_DOC_THRESHOLD:
        st.info("Content is substantial. Proceeding with AI analysis...")

        tabs = st.tabs(["📄 Summary", "📘 Glossary", "❓ Q&A", "🧠 MCQ"])

        # --- Summarization Logic ---
        with tabs[0]: # Summary Tab
            st.subheader("Topic-wise Summary")
            if st.session_state.summary_text is None:
                summary_spinner = st.empty()
                summary_spinner.info("Generating comprehensive summary...")
                try:
                    summary_response = requests.post(f"{BACKEND_URL}/summarize_document/", json={"text": full_text})
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        st.session_state.summary_text = summary_data.get("summary", "")
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

            glossary_spinner_placeholder = st.empty()

            # Only generate glossary if not already in session state or if it was empty and content exists
            should_generate_glossary = (
                st.session_state.glossary_content_string is None or
                (st.session_state.get('glossary_content_string') == "" and full_text)
            )

            if should_generate_glossary:
                glossary_spinner_placeholder.info("Generating technical glossary...")
                try:
                    glossary_response = requests.post(f"{BACKEND_URL}/generate_glossary/", json={"text": full_text})

                    if glossary_response.status_code == 200:
                        glossary_data = glossary_response.json()
                        generated_glossary_str = glossary_data.get("glossary", "")

                        st.session_state.glossary_content_string = generated_glossary_str
                        glossary_spinner_placeholder.empty()

                        if st.session_state.glossary_content_string:
                            st.success(f"Glossary generated successfully!")
                            st.markdown(st.session_state.glossary_content_string)
                        else:
                            st.warning("No glossary could be generated for this document.")
                    else:
                        glossary_spinner_placeholder.error(f"Error generating glossary: {glossary_response.status_code} - {glossary_response.text}")
                        st.session_state.glossary_content_string = ""
                except requests.exceptions.ConnectionError:
                    glossary_spinner_placeholder.error(f"Could not connect to the backend server for glossary generation at {BACKEND_URL}.")
                    st.session_state.glossary_content_string = ""
                except Exception as e:
                    glossary_spinner_placeholder.error(f"An unexpected error occurred during glossary generation: {e}")
                    st.session_state.glossary_content_string = ""

            elif st.session_state.get('glossary_content_string'):
                st.markdown(st.session_state.glossary_content_string)
            else:
                st.warning("No glossary could be generated yet. Please ensure the content contains technical terms.")

        # --- Q&A Logic ---
        with tabs[2]: # Q&A Tab
            st.subheader("Self-Testing Questions & Answers")
            if st.session_state.qa_pairs is None:
                qa_spinner = st.empty()
                qa_spinner.info("Generating Q&A pairs... This may take a moment.")
                try:
                    qa_response = requests.post(f"{BACKEND_URL}/generate_question_and_answer/", json={"text": full_text})
                    if qa_response.status_code == 200:
                        qa_data = qa_response.json()
                        st.session_state.qa_pairs = qa_data.get("qa_pairs", [])
                        qa_spinner.empty()
                        if st.session_state.qa_pairs:
                            st.success(f"Generated {len(st.session_state.qa_pairs)} Q&A pairs!")
                            for i, qa in enumerate(st.session_state.qa_pairs):
                                st.markdown(f"**Question {i+1}:** {qa.get('question', 'N/A')}")
                                with st.expander(f"Show Answer for Question {i+1}"):
                                    st.write(qa.get('answer', 'N/A'))
                                st.markdown("---")
                        else:
                            st.warning("No Q&A pairs could be generated for this content.")
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

        # --- MCQ Logic ---
        with tabs[3]: # MCQ Tab
            st.subheader("Multiple Choice Questions (MCQs) - Test Your Knowledge!")

            if not st.session_state.all_mcq_questions:
                mcq_spinner = st.empty()
                mcq_spinner.info("Generating ALL possible MCQs from content... This may take a moment.")
                try:
                    mcq_response = requests.post(
                        f"{BACKEND_URL}/generate_mcq/",
                        json={"text": full_text}
                    )
                    if mcq_response.status_code == 200:
                        mcq_data = mcq_response.json()
                        st.session_state.all_mcq_questions = mcq_data.get("mcqs", [])
                        mcq_spinner.empty()
                        if st.session_state.all_mcq_questions:
                            st.success(f"Generated {len(st.session_state.all_mcq_questions)} total MCQs! Preparing your test...")
                            initialize_new_test_instance(st.session_state.all_mcq_questions, NUM_QUESTIONS_PER_TEST)
                        else:
                            st.warning("No MCQs could be generated for this content.")
                    else:
                        mcq_spinner.error(f"Error generating MCQs: {mcq_response.status_code} - {mcq_response.text}")
                except requests.exceptions.ConnectionError:
                    mcq_spinner.error(f"Could not connect to the backend server for MCQ generation at {BACKEND_URL}.")
                except Exception as e:
                    mcq_spinner.error(f"An unexpected error occurred during MCQ generation: {e}")

            if st.session_state.all_mcq_questions:
                if not st.session_state.current_test_mcqs and not st.session_state.test_submitted:
                    initialize_new_test_instance(st.session_state.all_mcq_questions, NUM_QUESTIONS_PER_TEST)

                if st.session_state.current_test_mcqs:
                    if not st.session_state.test_submitted:
                        st.info(f"Answer the {len(st.session_state.current_test_mcqs)} questions below.")

                        for i, mcq_item in enumerate(st.session_state.current_test_mcqs):
                            st.markdown(f"**Question {i+1}:** {mcq_item.get('question', 'N/A')}")

                            shuffled_options = st.session_state.shuffled_options_map.get(i, [])

                            current_selection_index = None
                            if st.session_state.user_answers.get(i) in shuffled_options:
                                current_selection_index = shuffled_options.index(st.session_state.user_answers.get(i))

                            selected_option = st.radio(
                                f"Select your answer for Q{i+1} (Test ID: {st.session_state.test_instance_id}):",
                                shuffled_options,
                                index=current_selection_index,
                                key=f"mcq_q_{st.session_state.test_instance_id}_{i}"
                            )

                            st.session_state.user_answers[i] = selected_option
                            st.markdown("---")

                        if st.button("Submit Test for Scoring", key="submit_mcq_test"):
                            st.session_state.test_submitted = True
                            st.rerun()

                    else:
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
                                retest_current_instance()
                        with col2:
                            if st.button("Take New Test (Different Questions)", key="new_test"):
                                initialize_new_test_instance(st.session_state.all_mcq_questions, NUM_QUESTIONS_PER_TEST)

                else:
                    st.warning("No questions are currently loaded for the test. Click below to start.")
                    if st.button(f"Start New Test ({NUM_QUESTIONS_PER_TEST} Questions)", key="start_initial_test"):
                        initialize_new_test_instance(st.session_state.all_mcq_questions, NUM_QUESTIONS_PER_TEST)

            else:
                st.info("Upload a PDF, provide a YouTube URL, or paste text with substantial content to generate MCQs.")
    else:
        st.warning("The content is too short to generate a meaningful summary, glossary, Q&A, or MCQs.")
        st.info("Please provide content with more substantial text (at least 1000 characters) for best results.")

elif (uploaded_file is not None or youtube_url_input or text_input) and not full_text:
    st.error("Failed to process the provided content. Please check the error messages above for details.")
else:
    st.info("Upload a PDF file, provide a YouTube URL, or paste text in the box above to begin.")

st.sidebar.header("About ScholarMate")
st.sidebar.info(
    "ScholarMate helps you quickly grasp the essence of academic papers "
    "and technical documents by providing concise summaries, a "
    "technical glossary, self-testing Q&A sets, and multiple-choice questions, "
    "all powered by AI."
)
st.sidebar.markdown("---")
st.sidebar.caption("Developed with Streamlit, FastAPI, LangChain, and Groq.")