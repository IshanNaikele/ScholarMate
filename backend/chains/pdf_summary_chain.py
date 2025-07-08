from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langserve import add_routes 
from fastapi import FastAPI
import os

# Load environment variables from .env file at the very beginning
load_dotenv()

# Retrieve GROQ API key and ensure it's available
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the ChatGroq model
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
 
# 1. Create prompt template
# The user message will contain the actual content to be summarized.
prompt_template = ChatPromptTemplate.from_messages([
    ('system', "You are an expert academic summarizer. Summarize the following academic content concisely and accurately, focusing on key findings, methodologies, and conclusions. Maintain a neutral and objective tone."),
    ('user', '{input_text}') # {input_text} will be replaced by the content from your application
])

# Initialize the string output parser
parser = StrOutputParser()

# Create the LangChain chain: Prompt -> Model -> Parser
chain = prompt_template | llm | parser

 
 