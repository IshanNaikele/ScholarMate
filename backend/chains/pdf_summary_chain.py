# backend/chains/pdf_summary_chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# For long document summarization
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Represents a piece of text with metadata

# Load environment variables from .env file at the very beginning
load_dotenv()

# Retrieve GROQ API key and ensure it's available
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the ChatGroq model with a low temperature for consistent summaries
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key, temperature=0.1)

# --- Existing Chain (can be used for short inputs if needed) ---
# 1. Create prompt template for short summaries
prompt_template = ChatPromptTemplate.from_messages([
    ('system', "You are an expert academic summarizer. Summarize the following academic content concisely and accurately, focusing on key findings, methodologies, and conclusions. Maintain a neutral and objective tone."),
    ('user', '{input_text}')
])

# Initialize the string output parser
parser = StrOutputParser()

# Create the LangChain chain: Prompt -> Model -> Parser
# This 'chain' is suitable for inputs that fit within the LLM's context window.
chain = prompt_template | llm | parser

# --- New Function for Long Document Summarization ---
def summarize_long_document(full_text: str) -> str:
    """
    Summarizes a long document by chunking it and using LangChain's map_reduce summarization strategy.

    Args:
        full_text: The complete text of the document to be summarized.

    Returns:
        A concise summary of the entire document.
    """
    # Define the text splitter
    # chunk_size: Aim for a size well within the LLM's context window (Gemma2-9b-It is 8192 tokens).
    # 4000 characters is roughly 1000 tokens, leaving plenty of room for prompts and output.
    # chunk_overlap: Ensures context is not lost at chunk boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, # characters
        chunk_overlap=200, # characters
        length_function=len,
        is_separator_regex=False,
    )

    # Split the document into LangChain Document objects
    # create_documents takes a list of strings and returns a list of Document objects.
    chunks = text_splitter.create_documents([full_text])

    # Define prompts for the map and reduce steps
    map_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert academic summarizer. Summarize the following text snippet concisely, focusing on its main points. Keep it to 2-3 sentences."),
        ("user", "{text}")
    ])

    reduce_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert academic summarizer. Combine the following summaries into a single, cohesive, and comprehensive summary of the entire document. Focus on key findings, methodologies, and conclusions. The final summary should be approximately 300-500 words."),
        ("user", "{text}") # {text} here will contain the concatenated summaries from the map step
    ])

    # Initialize the summarization chain with 'map_reduce' strategy
    # verbose=True helps in debugging by showing the steps of the chain in the console.
    summarization_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,      # Use custom prompt for map step
        combine_prompt=reduce_prompt_template, # Use custom prompt for combine/reduce step
        verbose=True
    )

    # Run the summarization chain on the chunks
    final_summary = summarization_chain.run(chunks)

    return final_summary

# Example Usage (for testing this file directly, won't run via FastAPI)
if __name__ == "__main__":
    # Create a long example text for demonstration
    long_document_example = """
     3.1 Encoder and Decoder Stacks
 Encoder 
 32 5.01 25.4 60
 (C)
 2 6.11 23.7 36
 4 5.19 25.3 50
 8 4.88 25.5 80
 256 32 32 5.75 24.5 28
 1024 128 128 4.66 26.0 168
 1024 twork Grammar [8].
 In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the Berkeley
Parser [29] even when training only on the WSJ training set of 40K sentences.
    """  # Repeat to make it artificially long (e.g., 5000 words +)

    print("Starting long document summarization example...")
    try:
        summary = summarize_long_document(long_document_example)
        print("\n--- Long Document Summary ---")
        print(summary)
    except Exception as e:
        print(f"An error occurred during long document summarization: {e}")
        print("Please ensure your GROQ_API_KEY is correctly set and your LLM model is accessible.")