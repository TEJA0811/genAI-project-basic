import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def clean_text(raw_text):
    """
    Cleans extracted text to remove headers, footers, and line breaks mid-sentence.
    """
    cleaned = re.sub(r'\s+', ' ', raw_text)  # Remove excessive spaces/newlines
    cleaned = re.sub(r'Page \d+ of \d+', '', cleaned, flags=re.IGNORECASE)  # Remove headers/footers
    cleaned = re.sub(r'(\w)-\n(\w)', r'\1\2', cleaned)  # Remove mid-sentence line breaks
    cleaned = re.sub(r'\n', ' ', cleaned)  # Remove stray line breaks
    return cleaned

def split_text(cleaned_text, filename):
    """
    Splits text into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(cleaned_text)
    docs = [
        Document(page_content=chunk, metadata={"source": filename})
        for chunk in chunks
    ]
    return docs
