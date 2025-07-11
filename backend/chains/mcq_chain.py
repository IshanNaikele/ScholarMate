# chains/mcq_chain.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import re
import json # Import json for structured output parsing

# Load environment variables
load_dotenv()

# Retrieve GROQ API key and ensure it's available
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the ChatGroq model
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key, temperature=0.5) # Increased temperature slightly for creativity in options

def get_mcq_questions(full_text: str) -> list[dict]:
    """
    Generates Multiple Choice Questions (MCQs) from the given text using LangChain's map_reduce strategy.

    Args:
        full_text (str): The input text to generate MCQs from.

    Returns:
        list[dict]: A list of dictionaries, where each dict has 'question', 'options' (list), and 'correct_answer' (str).
    """
    if not full_text.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.create_documents([full_text])

    # --- Prompts for the Map and Reduce Steps for MCQ Generation ---

    # Map Prompt: Generate MCQs for each individual chunk
    # Instruct the LLM to provide only MCQs, clearly formatted.
    # Emphasize JSON output for easier parsing.
    mcq_map_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert educator generating concise multiple choice questions for self-testing.
        Your goal is to extract key information and concepts from the provided text and formulate them into multiple choice questions.
        Adhere to the following rules strictly:
        1. **Source ONLY:** All questions and options *must* be derived solely from the provided text. Do not use external knowledge for answers. Distractors should be plausible but incorrect based on the text.
        2. **Conciseness:** Questions should be direct, clear, and focused on one concept.
        3. **Quantity:** Generate 2-3 distinct multiple-choice questions per text chunk.
        4. **Format:** Each question MUST have exactly 4 options (A, B, C, D). One option must be correct, and the other three must be plausible incorrect distractors.
        5. **Output Structure:** Provide the output as a JSON array of objects. Each object should have the following keys:
           "question": "The question text",
           "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
           "correct_answer": "The exact text of the correct option"
        
        Example JSON output:
        [
            {{
                "question": "What is the capital of France?",
                "options": ["Berlin", "Madrid", "Paris", "Rome"],
                "correct_answer": "Paris"
            }},
            {{
                "question": "Which planet is known as the Red Planet?",
                "options": ["Earth", "Mars", "Jupiter", "Venus"],
                "correct_answer": "Mars"
            }}
        ]
        """),
        ("user", "{text}") # {text} here will be an individual chunk
    ])

    # Reduce Prompt: Consolidate and refine all MCQs generated from the map step.
    # This step is crucial for deduplication, ensuring overall quality, and maintaining JSON formatting.
    mcq_reduce_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic assistant. Your task is to compile a final, high-quality list of Multiple Choice Questions from the provided collection.
        Ensure the following strictly:
        1. **Deduplication:** Identify and remove duplicate or very similar questions. Prioritize the clearest version if minor variations exist.
        2. **Consistency:** Maintain a consistent, clear, and concise style for all questions and options.
        3. **Source Adherence:** Ensure all questions and correct answers are directly supported by the original text (as reflected in the input questions).
        4. **Format:** The final output MUST be a valid JSON array of objects. Each object must have "question", "options" (list of 4 strings), and "correct_answer" (exact string of the correct option).
        5. **Header (Optional):** Do NOT include any markdown headers or introductory text like '## Self-Testing Questions'. Just output the JSON array.
        6. **Order:** Present the questions in a logical flow, if possible, otherwise as they appear in the consolidated list.
        
        Here are the JSON formatted questions generated from chunks to consolidate (can be multiple JSON arrays, concatenate them):
        """),
        ("user", "{text}") # {text} here will contain the concatenated raw JSON from the map step
    ])

    # Initialize the summarize chain with 'map_reduce' strategy
    mcq_generation_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=mcq_map_prompt_template,
        combine_prompt=mcq_reduce_prompt_template,
        verbose=True # Keep verbose=True for debugging
    )

    # Run the chain on the document chunks
    raw_mcq_output = mcq_generation_chain.run(chunks)

    # Post-process the final string output from the LLM into a list of dictionaries.
    parsed_mcq_list = parse_mcq_json_output(raw_mcq_output)

    return parsed_mcq_list

def parse_mcq_json_output(json_output: str) -> list[dict]:
    """
    Parses the JSON formatted MCQ output from the LLM into a list of dictionaries.
    Assumes the output is a JSON array of objects.
    """
    mcq_list = []
    try:
        # Sometimes the LLM might include some preamble/postamble or wrap JSON in markdown code block.
        # Try to extract pure JSON.
        match = re.search(r"```json\s*(.*?)\s*```", json_output, re.DOTALL)
        if match:
            json_string = match.group(1).strip()
        else:
            json_string = json_output.strip()

        # Handle cases where multiple JSON arrays might be concatenated without proper delimiters
        # This is a common LLM issue. We'll try to find all valid JSON arrays.
        all_mcqs = []
        # Find all top-level JSON arrays
        json_array_matches = re.findall(r'\[\s*\{.*?\}\s*\]', json_string, re.DOTALL)
        for j_match in json_array_matches:
            try:
                parsed_block = json.loads(j_match)
                if isinstance(parsed_block, list):
                    all_mcqs.extend(parsed_block)
            except json.JSONDecodeError:
                # If a block isn't a perfect JSON array, try parsing it as a single object or line by line
                pass
        
        if not all_mcqs: # If no top-level arrays were found, try parsing the whole string directly
            try:
                all_mcqs = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"Primary JSON parse error: {e}. Attempting recovery...")
                # Attempt a more robust but less strict parsing if direct load fails
                # This is a fallback and might not always work perfectly.
                # Example: try to split by '},{' and wrap individual objects if necessary.
                
                # A common issue is concatenated objects like {..}{..}. Let's split by '}{'
                if '}{' in json_string:
                    parts = json_string.replace('}{', '},{').split('},{')
                    for part in parts:
                        try:
                            # Wrap in [] to make it a list for json.loads, assuming it's a single dict
                            if not part.strip().startswith('[') and not part.strip().endswith(']'):
                                part = '[' + part + ']'
                            temp_parsed = json.loads(part)
                            if isinstance(temp_parsed, list):
                                all_mcqs.extend(temp_parsed)
                            elif isinstance(temp_parsed, dict): # If it parsed a single dict
                                all_mcqs.append(temp_parsed)
                        except json.JSONDecodeError:
                            continue # Skip malformed parts
        
        # Validate each parsed MCQ structure
        for mcq in all_mcqs:
            if all(key in mcq for key in ["question", "options", "correct_answer"]) \
               and isinstance(mcq["question"], str) \
               and isinstance(mcq["options"], list) and len(mcq["options"]) == 4 \
               and isinstance(mcq["correct_answer"], str) \
               and mcq["correct_answer"] in mcq["options"]: # Ensure correct answer is one of the options
                mcq_list.append(mcq)
            else:
                print(f"Skipping malformed MCQ: {mcq}")

    except json.JSONDecodeError as e:
        print(f"Fatal JSON parsing error: {e}\nProblematic output: '{json_output}'")
        return [] # Return empty list on fatal error
    except Exception as e:
        print(f"An unexpected error occurred during MCQ parsing: {e}")
        return []

    return mcq_list


if __name__ == "__main__":
    # Example usage for testing
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    unlike the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents":
    any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    John McCarthy, who coined the term "artificial intelligence" in 1956, defined it as "the science and engineering of making intelligent machines".
    Machine learning (ML) is a subset of AI where systems learn from data without explicit programming.
    A common example is spam detection, where an ML model learns to identify spam emails based on features extracted from past emails.

    In natural language processing, the Transformer model has revolutionized sequence transduction tasks.
    It uses a novel attention mechanism, specifically Multi-Head Self-Attention, to weigh the importance
    of different words in a sentence. Unlike Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs),
    the Transformer eschews recurrence entirely, relying instead on positional encoding to inject word order.
    The architecture comprises an encoder that maps an input sequence of symbol representations
    (like word embeddings) to a sequence of continuous representations. The decoder then generates
    an output sequence one symbol at a time. Both encoder and decoder layers contain Position-wise Feed-Forward Networks
    and utilize Residual Connections followed by Layer Normalization. The decoder also employs Encoder-Decoder Attention.
    """
    print("Generating MCQs from sample text...")
    mcq_results = get_mcq_questions(sample_text) # Renamed function call

    if mcq_results:
        print("\n--- Generated MCQs ---")
        for i, mcq in enumerate(mcq_results):
            print(f"**Question {i+1}:** {mcq['question']}")
            for j, option in enumerate(mcq['options']):
                print(f"  {chr(65+j)}) {option}") # Prints A), B), C), D)
            print(f"  **Correct Answer:** {mcq['correct_answer']}")
            print("-" * 30)
    else:
        print("No MCQs generated.")