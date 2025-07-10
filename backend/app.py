# backend/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException # Added UploadFile, File, HTTPException
from langserve import add_routes
from dotenv import load_dotenv
import os
import uuid # To generate unique filenames for uploaded files
import sys
from pydantic import BaseModel

# --- Environment Variable Loading ---
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of app.py (backend/)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) # ScholarMate/
DATA_DIR = os.path.join(PROJECT_ROOT, "data") # ScholarMate/data/

os.makedirs(DATA_DIR, exist_ok=True)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- LangChain Chain Import ---
# Assuming 'chain' is defined in pdf_summary_chain.py
from backend.chains.pdf_summary_chain import chain as pdf_summary_chain_short
from backend.chains.pdf_summary_chain import summarize_long_document

from backend.chains.glossary_chain import chain as glossary_chain_short
from backend.chains.glossary_chain import get_glossary
# --- FastAPI Application Definition ---
app = FastAPI(
    title="ScholarMate Backend API",
    version="1.0",
    description="API for ScholarMate application, powered by LangChain and LangServe.",
)

# --- Define Project Root and Data Directory ---
# This ensures files are saved in the 'data' folder at the project's root,
# regardless of where app.py is run from.
 
# Ensure the data directory exists
 
# --- Import PDF Loader after path setup ---
# This import should now work correctly due to the __init__.py files
# and the project structure.
from backend.utils.pdf_loader import extract_text_from_pdf

# --- Root Endpoint for Health Check ---
@app.get("/")
async def root():
    return {"message": "ScholarMate Backend API is running!"}

# --- LangServe Route Addition ---
add_routes(
    app,
    pdf_summary_chain_short, # This uses the 'chain' from pdf_summary_chain.py
    path="/pdf-summary-short", # Renamed path for clarity
)

# --- New Endpoint for PDF Text Extraction ---
@app.post("/extract_text/")
async def get_text_from_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_location = os.path.join(DATA_DIR, unique_filename)

    try:
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        extracted_text = extract_text_from_pdf(file_location)
        if not extracted_text:
            raise HTTPException(status_code=500, detail="Could not extract text from PDF. The PDF might be empty or malformed.")

        return {"text": extracted_text}

    except Exception as e:
        print(f"Error processing PDF for text extraction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during text extraction: {str(e)}")
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

class SummarizeRequest(BaseModel):
    """Pydantic model for the request body of the summarization endpoint."""
    text: str # This field will receive the full extracted text from the frontend

@app.post("/summarize_document/")
async def summarize_document_endpoint(request: SummarizeRequest):
    """
    Receives long text and returns a comprehensive summary using LangChain's map-reduce.
    """
    try:
        # Call the long document summarization function
        summary = summarize_long_document(request.text)
        return {"summary": summary}
    except Exception as e:
        print(f"Error during long document summarization: {e}") # Log error on server side
        raise HTTPException(status_code=500, detail=f"An error occurred during summarization: {str(e)}")

# New: Pydantic model for the glossary request body
class GlossaryRequest(BaseModel):
    """Pydantic model for the request body of the glossary generation endpoint."""
    text: str # This field will receive the full extracted text from the frontend

# New: Endpoint for Glossary Generation
@app.post("/generate_glossary/")
async def generate_glossary_endpoint(request: GlossaryRequest):
    """
    Receives text and returns a technical glossary using LangChain's map-reduce.
    """
    try:
        glossary = get_glossary(request.text)
        return {"glossary": glossary}
    except Exception as e:
        print(f"Error during glossary generation: {e}") # Log error on server side
        raise HTTPException(status_code=500, detail=f"An error occurred during glossary generation: {str(e)}")


# --- Uvicorn Entry Point (for local development) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)