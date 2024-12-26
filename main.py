import os
import logging
import streamlit as st
import time
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import langchain
import shutil
import re

# Enable debugging
langchain.debug = True

# Configure logging to Streamlit
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("langchain_debug")
file_handler = logging.FileHandler("langchain_debug.log")
logger.addHandler(file_handler)

file_path = "C:\\Users\\tejas\\Desktop\\myDoc\\Major project-report.pdf"

load_dotenv()  # Load environment variables

st.title("Unlock the wisdom in your data.")
st.sidebar.title("File Upload")

# Initialize session state for db and chain
if "db" not in st.session_state:
    st.session_state.db = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

process_file_clicked = st.sidebar.button("Upload")

main_placeholder = st.empty()
llm = ChatOpenAI(model="gpt-4o-mini")

# Function for cleaning raw text
def clean_text(raw_text):
    """
    Cleans extracted text to remove headers, footers, and line breaks mid-sentence.
    """
    # Remove excessive spaces and newlines
    cleaned = re.sub(r'\s+', ' ', raw_text)
    # Remove headers/footers if they follow a pattern (e.g., "Page X of Y")
    cleaned = re.sub(r'Page \d+ of \d+', '', cleaned, flags=re.IGNORECASE)
    # Remove mid-sentence line breaks caused by PDF formatting
    cleaned = re.sub(r'(\w)-\n(\w)', r'\1\2', cleaned)
    # Remove stray line breaks
    cleaned = re.sub(r'\n', ' ', cleaned)
    return cleaned

if process_file_clicked:
    # Reset session state for a fresh start
    st.session_state.db = None
    st.session_state.qa_chain = None
    main_placeholder.text("Processing uploaded file...✅✅✅")

    # Load PDF data
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # Synchronous load
    logger.debug(f"Data loaded: {pages}")
    logger.debug(f"Number of pages loaded: {len(pages)}")

    # Clean and concatenate pages
    main_placeholder.text("Cleaning extracted text...✅✅✅")
    cleaned_pages = [clean_text(page.page_content) for page in pages]
    cleaned_text = " ".join(cleaned_pages)
    logger.debug(f"Cleaned text: {cleaned_text[:1000]}")  # Log first 1000 characters for inspection

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=500,
        chunk_overlap=50  # Add overlap for better context continuity
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    chunks = text_splitter.split_text(cleaned_text)
    docs = [
        Document(page_content=chunk, metadata={"source": file_path})
        for chunk in chunks
    ]

    logger.debug(f"Documents with metadata: {docs[:5]}")  # Log first 5 docs for inspection
    logger.debug(f"Documents after splitting: {docs}")

    # Create embeddings and rebuild FAISS index
    if docs:
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Remove old FAISS index file if exists
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")  # Removes the entire directory
            logger.debug("Old FAISS index directory removed.")

        # Create new FAISS index
        # Create FAISS index
        st.session_state.db = FAISS.from_documents(docs, OpenAIEmbeddings())
        logger.debug(f"FAISS index created successfully: {st.session_state.db}")

        # Save the FAISS index for future use (optional)
        st.session_state.db.save_local("faiss_index")

        # Create the RetrievalQAWithSourcesChain
        st.session_state.qa_chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=st.session_state.db.as_retriever(),
            verbose=True  # Enable verbose mode for the chain
        )
        logger.debug(f"QA Chain created successfully: {st.session_state.qa_chain}")
    else:
        main_placeholder.text("No documents to create FAISS index.")
        st.session_state.db = None
        st.session_state.qa_chain = None

query = main_placeholder.text_input("Question: ")

if query:
    if st.session_state.qa_chain is not None:
        # Use the QA chain to get the answer with sources
        result = st.session_state.qa_chain({"question": query}, return_only_outputs=True)
        logger.debug(f"Query Result: {result}")

        # Display the result
        st.header("Answer")
        st.write(result["answer"] if "answer" in result else "No answer found.")

        # Display sources
        if "sources" in result:
            st.subheader("Sources:")
            sources = result["sources"].split("\n")  # Split sources by newline
            for source in sources:
                st.write(source)
    else:
        st.error("Please upload the file first by clicking 'Upload'.")
