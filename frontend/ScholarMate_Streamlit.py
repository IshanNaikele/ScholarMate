# frontend/ScholarMate_Streamlit.py

import streamlit as st
import requests
import os
import sys
import random # For shuffling MCQ options

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

        # Initialize outputs (though largely handled by session_state now)
        summary_output = ""
        glossary_output = ""
        qa_pairs_output = []
        mcq_output = [] 

        if len(extracted_text) > LONG_DOC_THRESHOLD:
            st.info("Document content is substantial. Proceeding with AI analysis...")

            # Create tabs for Summary, Glossary, Q&A, and MCQ
            tabs = st.tabs(["📄 Summary", "📘 Glossary", "❓ Q&A", "🧠 MCQ"])

            # --- Summarization Logic ---
            # with tabs[0]: # Summary Tab
            #     st.subheader("Topic-wise Summary")
            #     # Initialize summary in session state to avoid re-generating on every rerun
            #     if 'summary_text' not in st.session_state or st.session_state.get('last_extracted_text') != extracted_text:
            #         summary_spinner = st.empty()
            #         summary_spinner.info("Generating comprehensive summary...")
            #         try:
            #             summary_response = requests.post(
            #                 f"{BACKEND_URL}/summarize_document/",
            #                 json={"text": extracted_text}
            #             )
            #             if summary_response.status_code == 200:
            #                 summary_data = summary_response.json()
            #                 st.session_state.summary_text = summary_data.get("summary", "")
            #                 st.session_state.last_extracted_text = extracted_text # Store text to prevent re-gen
            #                 summary_spinner.empty() # Clear spinner on success
            #                 if st.session_state.summary_text:
            #                     st.markdown(st.session_state.summary_text)
            #                 else:
            #                     st.warning("No comprehensive summary could be generated.")
            #             else:
            #                 summary_spinner.error(f"Error generating summary: {summary_response.status_code} - {summary_response.text}")
            #         except requests.exceptions.ConnectionError:
            #             summary_spinner.error(f"Could not connect to the backend server for summarization at {BACKEND_URL}.")
            #         except Exception as e:
            #             summary_spinner.error(f"An unexpected error occurred during summarization: {e}")
            #     elif st.session_state.summary_text:
            #         st.markdown(st.session_state.summary_text) # Display existing summary
            #     else:
            #         st.warning("No comprehensive summary could be generated yet.")


            # # --- Glossary Logic ---
            # with tabs[1]: # Glossary Tab
            #     st.subheader("Technical Glossary (Glassador)")
            #     # Initialize glossary in session state
            #     if 'glossary_terms' not in st.session_state or st.session_state.get('last_extracted_text_glossary') != extracted_text:
            #         glossary_spinner = st.empty()
            #         glossary_spinner.info("Generating technical glossary...")
            #         try:
            #             glossary_response = requests.post(
            #                 f"{BACKEND_URL}/generate_glossary/",
            #                 json={"text": extracted_text}
            #             )
            #             if glossary_response.status_code == 200:
            #                 glossary_data = glossary_response.json()
            #                 st.session_state.glossary_terms = glossary_data.get("glossary", []) 
            #                 st.session_state.last_extracted_text_glossary = extracted_text
            #                 glossary_spinner.empty() # Clear spinner on success
            #                 if st.session_state.glossary_terms:
            #                     for item in st.session_state.glossary_terms:
            #                         if isinstance(item, dict) and "term" in item and "definition" in item:
            #                             st.markdown(f"**{item['term']}**: {item['definition']}")
            #                         elif isinstance(item, str):
            #                             st.markdown(item)
            #                     st.success(f"Generated {len(st.session_state.glossary_terms)} glossary terms!")
            #                 else:
            #                     st.warning("No glossary available.")
            #             else:
            #                 glossary_spinner.error(f"Error generating glossary: {glossary_response.status_code} - {glossary_response.text}")
            #         except requests.exceptions.ConnectionError:
            #             glossary_spinner.error(f"Could not connect to the backend server for glossary generation at {BACKEND_URL}.")
            #         except Exception as e:
            #             glossary_spinner.error(f"An unexpected error occurred during glossary generation: {e}")
            #     elif st.session_state.glossary_terms:
            #         for item in st.session_state.glossary_terms:
            #             if isinstance(item, dict) and "term" in item and "definition" in item:
            #                 st.markdown(f"**{item['term']}**: {item['definition']}")
            #             elif isinstance(item, str):
            #                 st.markdown(item)
            #     else:
            #         st.warning("No glossary could be generated yet.")

            # # --- Q&A Logic ---
            # with tabs[2]: # Q&A Tab
            #     st.subheader("Self-Testing Questions & Answers")
            #     # Initialize Q&A in session state
            #     if 'qa_pairs' not in st.session_state or st.session_state.get('last_extracted_text_qa') != extracted_text:
            #         qa_spinner = st.empty()
            #         qa_spinner.info("Generating Q&A pairs... This may take a moment.")
            #         try:
            #             qa_response = requests.post(
            #                 f"{BACKEND_URL}/generate_question_and_answer/",
            #                 json={"text": extracted_text}
            #             )
            #             if qa_response.status_code == 200:
            #                 qa_data = qa_response.json()
            #                 st.session_state.qa_pairs = qa_data.get("qa_pairs", [])
            #                 st.session_state.last_extracted_text_qa = extracted_text
            #                 qa_spinner.empty()
            #                 if st.session_state.qa_pairs:
            #                     st.success(f"Generated {len(st.session_state.qa_pairs)} Q&A pairs!")
            #                     for i, qa in enumerate(st.session_state.qa_pairs):
            #                         st.markdown(f"**Question {i+1}:** {qa.get('question', 'N/A')}")
            #                         with st.expander(f"Show Answer for Question {i+1}"):
            #                             st.write(qa.get('answer', 'N/A'))
            #                         st.markdown("---")
            #                 else:
            #                     st.warning("No Q&A pairs could be generated for this document.")
            #             else:
            #                 qa_spinner.error(f"Error generating Q&A: {qa_response.status_code} - {qa_response.text}")
            #         except requests.exceptions.ConnectionError:
            #             qa_spinner.error(f"Could not connect to the backend server for Q&A generation at {BACKEND_URL}. Please ensure your FastAPI backend is running.")
            #         except Exception as e:
            #             qa_spinner.error(f"An unexpected error occurred during Q&A generation: {e}")
            #     elif st.session_state.qa_pairs:
            #         for i, qa in enumerate(st.session_state.qa_pairs):
            #             st.markdown(f"**Question {i+1}:** {qa.get('question', 'N/A')}")
            #             with st.expander(f"Show Answer for Question {i+1}"):
            #                 st.write(qa.get('answer', 'N/A'))
            #             st.markdown("---")
            #     else:
            #         st.warning("No Q&A pairs could be generated yet.")

            # --- MCQ Logic (NEW SECTION for Test) ---
            with tabs[3]: # MCQ Tab
                st.subheader("Multiple Choice Questions (MCQs) - Test Your Knowledge!")

                # Initialize session state for MCQs and user answers
                if 'mcq_questions' not in st.session_state or st.session_state.get('last_extracted_text_mcq') != extracted_text:
                    st.session_state.mcq_questions = []
                    st.session_state.user_answers = {}
                    st.session_state.test_submitted = False
                    st.session_state.last_extracted_text_mcq = extracted_text # Track text to avoid re-gen

                    mcq_spinner = st.empty()
                    mcq_spinner.info("Generating MCQs... This may take a moment.")
                    try:
                        mcq_response = requests.post(
                            f"{BACKEND_URL}/generate_mcq/",
                            json={"text": extracted_text}
                        )
                        if mcq_response.status_code == 200:
                            mcq_data = mcq_response.json()
                            st.session_state.mcq_questions = mcq_data.get("mcqs", [])
                            # Initialize user answers for each question
                            st.session_state.user_answers = {i: None for i in range(len(st.session_state.mcq_questions))}
                            
                            # Shuffle options once and store them
                            for i, mcq_item in enumerate(st.session_state.mcq_questions):
                                options = mcq_item.get('options', [])
                                if options:
                                    st.session_state[f'shuffled_options_{i}'] = random.sample(options, len(options))
                                else:
                                    st.session_state[f'shuffled_options_{i}'] = []

                            mcq_spinner.empty()
                            if st.session_state.mcq_questions:
                                st.success(f"Generated {len(st.session_state.mcq_questions)} MCQs! Please answer the questions below.")
                            else:
                                st.warning("No MCQs could be generated for this document.")
                        else:
                            mcq_spinner.error(f"Error generating MCQs: {mcq_response.status_code} - {mcq_response.text}")
                    except requests.exceptions.ConnectionError:
                        mcq_spinner.error(f"Could not connect to the backend server for MCQ generation at {BACKEND_URL}.")
                    except Exception as e:
                        mcq_spinner.error(f"An unexpected error occurred during MCQ generation: {e}")

                # Display MCQs and collect answers if questions exist
                if st.session_state.mcq_questions:
                    for i, mcq_item in enumerate(st.session_state.mcq_questions):
                        st.markdown(f"**Question {i+1}:** {mcq_item.get('question', 'N/A')}")
                        
                        shuffled_options = st.session_state.get(f'shuffled_options_{i}', [])
                        
                        # Use on_change to immediately update the user's answer in session state
                        def update_answer(index, selected_value):
                            st.session_state.user_answers[index] = selected_value

                        selected_option = st.radio(
                            f"Select your answer for Q{i+1}:",
                            shuffled_options,
                            index=shuffled_options.index(st.session_state.user_answers.get(i)) if st.session_state.user_answers.get(i) in shuffled_options else 0, # Keep previous selection if exists, else default 0
                            key=f"mcq_q_{i}",
                            on_change=update_answer,
                            args=(i, st.session_state.user_answers.get(i)), # Pass current value to on_change
                        )
                        # Ensure the latest selected option is always stored, especially if on_change doesn't fire immediately
                        st.session_state.user_answers[i] = selected_option
                        
                        st.markdown("---")

                    # Submit Button for the entire test
                    if not st.session_state.test_submitted:
                        if st.button("Submit Test for Scoring", key="submit_mcq_test"):
                            st.session_state.test_submitted = True
                            st.rerun() # Rerun to display results
                    else: # If test is already submitted, show results
                        total_questions = len(st.session_state.mcq_questions)
                        correct_count = 0
                        
                        st.subheader("Test Results:")
                        for i, mcq_item in enumerate(st.session_state.mcq_questions):
                            user_ans = st.session_state.user_answers.get(i)
                            correct_ans = mcq_item.get('correct_answer', 'N/A')

                            st.markdown(f"**Question {i+1}:** {mcq_item.get('question', 'N/A')}")
                            st.write(f"Your Answer: **{user_ans if user_ans is not None else 'No Answer'}**")
                            st.write(f"Correct Answer: **{correct_ans}**")
                            
                            # Case-insensitive comparison for robustness
                            if str(user_ans).strip().lower() == str(correct_ans).strip().lower():
                                st.success("🎉 Correct!")
                                correct_count += 1
                            else:
                                st.error("❌ Incorrect!")
                            st.markdown("---")
                        
                        st.subheader(f"Final Score: {correct_count} out of {total_questions}")
                        if total_questions > 0:
                            st.progress(correct_count / total_questions)
                        
                        if st.button("Retake Test / Generate New MCQs", key="retake_mcq_test"):
                            del st.session_state.mcq_questions
                            del st.session_state.user_answers
                            del st.session_state.test_submitted
                            del st.session_state.last_extracted_text_mcq
                            # Clear shuffled options from session state too
                            for i in range(total_questions):
                                if f'shuffled_options_{i}' in st.session_state:
                                    del st.session_state[f'shuffled_options_{i}']
                            st.rerun() # Force rerun to re-initialize or re-generate
                else:
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