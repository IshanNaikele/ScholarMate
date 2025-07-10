from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import re
# For long document summarization
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Represents a piece of text with metadata

# Load environment variables
load_dotenv()

# Retrieve GROQ API key and ensure it's available
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the ChatGroq model with a low temperature for consistent summaries
# Consider slightly increasing temperature (e.g., 0.3) for glossary generation
# to allow for more creative term identification and varied examples.
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key, temperature=0.2) # Changed temperature

# 1. Create prompt template for short summaries (This is for a different chain, just keeping for context)
prompt_template = ChatPromptTemplate.from_messages([
    ('system', "You are an expert academic summarizer. Summarize the following academic content concisely and accurately, focusing on key findings, methodologies, and conclusions. Maintain a neutral and objective tone."),
    ('user', '{input_text}')
])

# Initialize the string output parser
parser = StrOutputParser()

# Create the LangChain chain: Prompt -> Model -> Parser
# This 'chain' is suitable for inputs that fit within the LLM's context window.
chain = prompt_template | llm | parser


def get_glossary(full_text: str) -> str:
    """
    Generates a technical glossary from the given text using LangChain and an LLM.
    It attempts to identify key terms and provide definitions.

    Args:
        full_text (str): The input text to extract glossary terms from.

    Returns:
        str: The generated glossary in markdown format.
    """
    if not full_text.strip():
        return "No text provided for glossary."

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, # characters
        chunk_overlap=200, # characters
        length_function=len,
        is_separator_regex=False,
    )

    # Split the document into LangChain Document objects
    chunks = text_splitter.create_documents([full_text])

    # --- Refined Prompts for the Map and Reduce Steps ---
    map_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic assistant tasked with identifying and explaining technical terms from a text.
        For each technical term you identify, provide its concise definition and a relevant example.
        Prioritize examples found within the provided text. If no suitable example is in the text, provide a general, clear example.
        Format each term and its explanation as: 'Term: Explanation with example.'
        Ensure the term is exactly as it appears in the text where possible. Focus only on technical or specialized terms relevant to the domain."""),
        ("user", "{text}")
    ])

    reduce_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic assistant. Your task is to compile a final, comprehensive technical glossary from the provided list of terms and their explanations.
        Ensure the following:
        1. **Combine and Deduplicate:** Merge definitions for the same term and ensure each unique term appears only once.
        2. **Consistency:** Maintain a consistent, clear, and concise style for all definitions and examples.
        3. **Formatting:** Present the glossary as a well-structured markdown list.
        4. **Order:** Alphabetize the terms for easy readability.
        5. **Header:** Start the glossary with '## Technical Glossary'.
        6. **Clarity:** Avoid redundancy and ensure clarity in all explanations.
        
        Here are the terms and explanations to consolidate:
        """),
        ("user", "{text}") # {text} here will contain the concatenated explanations from the map step
    ])

    # Initialize the summarization chain with 'map_reduce' strategy
    glossary_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,        # Use custom prompt for map step
        combine_prompt=reduce_prompt_template, # Use custom prompt for combine/reduce step
        verbose=True # Keep verbose=True for debugging
    )

    # Run the summarization chain on the chunks
    final_glossary = glossary_chain.run(chunks)

    # Optional: Post-processing to ensure alphabetization and proper markdown if LLM output varies
    # This regex attempts to find lines starting with '- **Term**:'
    # entries = re.findall(r"^- \*\*(.*?)\*\*:.*$", final_glossary, re.MULTILINE)
    # if entries:
    #     # A simple way to re-alphabetize, though might lose original context if not careful
    #     # if the LLM didn't follow the instruction. It's better to rely on LLM.
    #     pass # For now, trust the LLM's alphabetical ordering from the prompt

    # Ensure a header is present if LLM missed it, though the prompt now asks for it.
    if not final_glossary.strip().startswith("## Technical Glossary"):
        final_glossary = "## Technical Glossary\n" + final_glossary

    return final_glossary

if __name__ == "__main__":
    # Example usage for testing
    sample_text = """
    Quantum computing leverages principles of quantum mechanics, such as superposition and entanglement,
    to perform computations. Unlike classical computers that use bits representing 0 or 1, quantum
    computers use qubits, which can represent 0, 1, or both simultaneously. This enables them to solve
    certain problems much faster than classical computers. A practical example is Shor's algorithm,
    which can factor large numbers exponentially faster than classical algorithms, potentially
    breaking current encryption methods.

    Blockchain technology is a decentralized, distributed ledger that records transactions across
    many computers. It's known for its security features, including cryptographic hashing, which
    ensures data integrity, and immutability, meaning records cannot be altered. Cryptocurrencies
    like Bitcoin are built on blockchain. For instance, in Bitcoin, every transaction is a block,
    and these blocks are linked together using cryptographic hashes, forming a chain. Smart contracts,
    self-executing contracts with the terms of the agreement directly written into code, also utilize blockchain.
    An example of a smart contract is an escrow agreement where funds are automatically released
    to a seller once delivery is confirmed by an oracle.

    Artificial intelligence (AI) is a broad field focused on creating machines that can
    reason, learn, and act intelligently. Machine learning (ML) is a subset of AI where
    systems learn from data without explicit programming. A common example is spam detection,
    where an ML model learns to identify spam emails based on features extracted from past emails.

    In natural language processing, the Transformer model has revolutionized sequence transduction tasks.
    It uses a novel attention mechanism, specifically Multi-Head Self-Attention, to weigh the importance
    of different words in a sentence. Unlike Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs),
    the Transformer eschews recurrence entirely, relying instead on positional encoding to inject word order.
    The architecture comprises an encoder that maps an input sequence of symbol representations
    (like word embeddings) to a sequence of continuous representations. The decoder then generates
    an output sequence one symbol at a time. Both encoder and decoder layers contain Position-wise Feed-Forward Networks
    and utilize Residual Connections followed by Layer Normalization. The decoder also employs Encoder-Decoder Attention.
    """
    glossary = get_glossary(sample_text)
    print("\n--- Generated Glossary ---")
    print(glossary)