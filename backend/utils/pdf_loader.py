import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A single string containing all extracted text.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

