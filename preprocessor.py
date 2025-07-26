# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import pymupdf

# def extract_text_from_pdf(pdf_path):
#     """Fast PDF text extraction (unchanged)"""
#     doc = pymupdf.open(pdf_path)
#     return " ".join(page.get_text() for page in doc)

# def clean_text(text):
#     """Minimal cleaning preserving Bengali punctuation"""
#     return text.replace("\n", " ").strip()

# def chunk_text(text, chunk_size=500, overlap=60):
#     """Improved chunking with Bangla-aware recursive splitting"""
#     # Create splitter with Bangla-specific settings
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=overlap,
#         separators=["\n\n", "\n", "।", "?", "!", ".", " ", ""],  # Bangla punctuation first
#         keep_separator=True,
#         is_separator_regex=False
#     )
    
#     return splitter.split_text(text)




from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document  # For wrapping text
import pymupdf  # Your current PDF loader

def extract_text_from_pdf(pdf_path):
    """Keep your existing fast PDF extraction."""
    doc = pymupdf.open(pdf_path)
    return " ".join(page.get_text() for page in doc)

def clean_text(text):
    """Minimal cleaning (preserves Bengali punctuation)."""
    return text.replace("\n", " ").strip()

def chunk_text(text, chunk_size=600, overlap=60):
    """Replace fixed-size chunks with LlamaIndex's Bangla-aware splitting."""
    # Wrap text in a LlamaIndex Document
    document = Document(text=text)
    
    # Initialize parser with Bengali support
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;।]+[,.;।]?",  # Handles Bengali punctuation
    )
    
    # Generate nodes and extract text
    nodes = splitter.get_nodes_from_documents([document])
    return [node.text for node in nodes]

