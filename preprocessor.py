from langdetect import detect
#import fitz  # PyMuPDF
import pymupdf

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    return " ".join(page.get_text() for page in doc)

def clean_text(text):
    # remove extra whitespace, newline, etc.
    return text.replace("\n", " ").strip()

def chunk_text(text, chunk_size=512, overlap=64):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# def chunk_text(text, chunk_size=512, overlap=64):
#     words = text.split()  # Split into words first
#     chunks = []
    
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = ' '.join(words[i:i + chunk_size])
#         chunks.append(chunk)
    
#     return chunks