# backend/app.py
from fastapi import FastAPI
from langserve import add_routes
from dotenv import load_dotenv # Used to load environment variables from .env
import os # Used to access environment variables

# --- Environment Variable Loading ---
# Load environment variables from .env file at the very beginning of the application startup.
# This ensures that API keys and other configurations are available before they are needed.
load_dotenv()

# Retrieve GROQ API key and ensure it's available.
# It's crucial to check for the API key here, as the application cannot function without it.
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    # Raise an error if the API key is not found, providing a clear message to the user.
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# --- LangChain Chain Import ---
# Import the 'chain' object from your pdf_summary_chain.py.
# The 'chain' variable from that file is aliased as 'pdf_summary_chain' for clarity.
from chains.pdf_summary_chain import chain as pdf_summary_chain

# --- FastAPI Application Definition ---
# Initialize the FastAPI application.
# Provide a title, version, and description for better API documentation (Swagger UI).
app = FastAPI(
    title="ScholarMate Backend API",
    version="1.0",
    description="API for ScholarMate application, powered by LangChain and LangServe.",
)

# --- Root Endpoint for Health Check ---
# Add a simple GET endpoint at the root URL to confirm the API is running.
@app.get("/")
async def root():
    return {"message": "ScholarMate Backend API is running!"}

# --- LangServe Route Addition ---
# Add routes for your LangChain chain to the FastAPI app using LangServe.
# This automatically creates several endpoints (e.g., /pdf-summary/invoke, /pdf-summary/stream)
# that allow your frontend to interact with the summarization chain.
add_routes(
    app,
    pdf_summary_chain,
    path="/pdf-summary", # The base path for this chain's API endpoints
    # You can optionally add a name for the route, which appears in the LangServe playground.
    # name="PDF Summary Chain",
)

# --- Uvicorn Entry Point (for local development) ---
# This block ensures that the Uvicorn server runs when app.py is executed directly.
# It's typically used for local development and testing.
if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app on localhost at port 8000.
    # The 'reload=True' argument enables auto-reloading on code changes, which is convenient for development.
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

