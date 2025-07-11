# chains/qa_chain.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain # Re-using this for map_reduce
from langchain_core.documents import Document # For creating Document objects
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Retrieve GROQ API key and ensure it's available
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the ChatGroq model
# For Q&A, a slightly higher temperature (e.g., 0.5-0.7) might encourage
# more varied questions, but start with 0.2-0.3 for consistency and accuracy.
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key, temperature=0.2)

def get_qa_pairs(full_text: str) -> list[dict]:
    """
    Generates Question-Answer pairs from the given text using LangChain's map_reduce strategy.

    Args:
        full_text (str): The input text to generate Q&A from.

    Returns:
        list[dict]: A list of dictionaries, where each dict has 'question' and 'answer' keys.
    """
    if not full_text.strip():
        return []

    # Text splitting - Use similar chunking as your glossary, but perhaps slightly smaller
    # for more focused Q&A generation per chunk.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Smaller chunks might lead to more focused Q&A
        chunk_overlap=100, # Overlap helps maintain context across chunks
        length_function=len,
        is_separator_regex=False,
    )

    # Convert the full text into LangChain Document objects
    # The load_summarize_chain expects a list of Document objects
    chunks = text_splitter.create_documents([full_text])

    # --- Prompts for the Map and Reduce Steps for Q&A Generation ---

    # Map Prompt: Generate Q&A for each individual chunk
    # Instruct the LLM to provide only Q&A pairs, clearly formatted.
    qa_map_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert educator generating concise question-answer pairs for self-testing.
        Your goal is to extract key information and concepts from the provided text and formulate them into questions and answers.
        Adhere to the following rules:
        1. **Source ONLY:** All questions and answers *must* be derived solely from the provided text. Do not use external knowledge.
        2. **Conciseness:** Answers should be direct and ideally 1-3 sentences long.
        3. **Clarity:** Questions and answers should be clear and easy to understand.
        4. **Quantity:** Generate 3-5 distinct question-answer pairs per text chunk.
        5. **Format:** Each pair must be formatted as:
           Question: [Your question here]
           Answer: [The corresponding answer from the text]
        """),
        ("user", "{text}") # {text} here will be an individual chunk
    ])

    # Reduce Prompt: Consolidate and refine all Q&A pairs generated from the map step.
    # This step is crucial for deduplication and ensuring overall quality and formatting.
    qa_reduce_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic assistant. Your task is to compile a final, high-quality list of Question-Answer pairs from the provided collection.
        Ensure the following:
        1. **Deduplication:** Identify and remove duplicate or very similar Q&A pairs. Prioritize the clearest or most comprehensive version if minor variations exist.
        2. **Consistency:** Maintain a consistent, clear, and concise style for all questions and answers.
        3. **Source Adherence:** Ensure all answers are directly supported by the original text (as reflected in the input pairs).
        4. **Formatting:** Present the final Q&A pairs as a markdown list. Each question should be bolded, followed by its answer.
           Example:
           **Question:** What is quantum computing?
           **Answer:** Quantum computing uses principles like superposition and entanglement for computation, unlike classical computers.

           **Question:** What is a qubit?
           **Answer:** A qubit is a unit of quantum information that can represent 0, 1, or both simultaneously.
        5. **Header:** Start the output with '## Self-Testing Questions & Answers'.
        6. **Order:** Present the Q&A pairs in a logical flow, if possible, otherwise by their appearance in the consolidated list.
        
        Here are the generated Q&A pairs to consolidate:
        """),
        ("user", "{text}") # {text} here will contain the concatenated raw Q&A from the map step
    ])

    # Initialize the summarize chain with 'map_reduce' strategy
    # The 'summarize' chain type is versatile and works for combining chunk-level outputs.
    qa_generation_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=qa_map_prompt_template,
        combine_prompt=qa_reduce_prompt_template,
        verbose=True # Keep verbose=True for debugging
    )

    # Run the chain on the document chunks
    raw_qa_output = qa_generation_chain.run(chunks)

    # Post-process the final string output from the LLM into a list of dictionaries.
    # This is similar to the parsing logic I provided earlier, adapted for the markdown output.
    parsed_qa_list = parse_qa_markdown_output(raw_qa_output)

    return parsed_qa_list

def parse_qa_markdown_output(markdown_output: str) -> list[dict]:
    """
    Parses the markdown formatted Q&A output from the LLM into a list of dictionaries.
    Assumes the format:
    **Question:** ...
    **Answer:** ...
    """
    qa_pairs = []
    
    # First, split the entire output by the "## Self-Testing Questions & Answers" header if present,
    # and then focus on the content after it. If not present, use the whole output.
    content_start_index = markdown_output.find("## Self-Testing Questions & Answers")
    if content_start_index != -1:
        # Add length of header to get to the actual content
        relevant_content = markdown_output[content_start_index + len("## Self-Testing Questions & Answers"):].strip()
    else:
        relevant_content = markdown_output.strip()

    # Now, split the relevant content by "**Question:" to get individual Q&A blocks.
    # The first split might be empty if the content starts directly with "**Question:".
    blocks = relevant_content.split("**Question:")
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue # Skip empty blocks (e.g., the part before the first "**Question:")

        # Each block should now start with the question content, then "**Answer:"
        if "**Answer:" in block:
            try:
                # Split the block into question and answer parts
                question_part, answer_part = block.split("**Answer:", 1)
                
                # Clean up the parts
                # The question part starts after "**Question:", so we need to trim the markdown.
                # It might have leading newlines from the split.
                question = question_part.strip().replace('\n', ' ')
                
                # The answer part starts after "**Answer:", so just strip and replace newlines.
                answer = answer_part.strip().replace('\n', ' ')
                
                # Remove any leftover markdown from the start of question/answer if present
                if question.startswith('**'):
                    question = question[2:].strip()
                if answer.startswith('**'):
                    answer = answer[2:].strip()

                if question and answer: # Ensure both parts are non-empty
                    qa_pairs.append({"question": question, "answer": answer})
            except Exception as e:
                print(f"Error parsing Q&A block: {e}\nProblematic block content: '{block}'")
                continue # Continue to the next block if parsing fails

    return qa_pairs


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
    print("Generating Q&A pairs from sample text...")
    qa_results = get_qa_pairs(sample_text)

    if qa_results:
        print("\n--- Generated Q&A Pairs ---")
        for i, qa in enumerate(qa_results):
            print(f"**Question {i+1}:** {qa['question']}")
            print(f"**Answer:** {qa['answer']}")
            print("-" * 30)
    else:
        print("No Q&A pairs generated.")