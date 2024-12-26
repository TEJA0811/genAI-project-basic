import os
import fitz
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.text_processing import clean_text, split_text
from app.faiss_utils import create_faiss_index
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from app.config import UPLOAD_DIR, FAISS_DIR
from dotenv import load_dotenv
from io import BytesIO
import pickle
import faiss

load_dotenv()

router = APIRouter()
llm = ChatOpenAI(model="gpt-4o-mini")
db = None

# Logic to load the FAISS index and pickled data if files exist
index_file = os.path.join(FAISS_DIR, "index.faiss")
pickle_file = os.path.join(FAISS_DIR, "index.pkl")

if os.path.exists(index_file) and os.path.exists(pickle_file):
    # Load the FAISS index
    db = faiss.read_index(index_file)

    # Load the pickled object (assuming it stores additional metadata or configurations)
    with open(pickle_file, 'rb') as f:
        db_pickle = pickle.load(f)
else:
    print("FAISS index or pickle file not found. Setting db to None.")


@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file and process it.
    """
    try:
        file_content = await file.read()
        file_stream = BytesIO(file_content)

        # Process PDF using PyMuPDF
        pdf_document = fitz.open(stream=file_stream, filetype="pdf")

        # Extract text from each page
        pages = [pdf_document[page_number].get_text("text") for page_number in range(len(pdf_document))]
        pdf_document.close()

        cleaned_pages = [clean_text(page) for page in pages]
        cleaned_text = " ".join(cleaned_pages)

        # Split text into chunks and add metadata with the `source` field
        docs = split_text(cleaned_text, file.filename)

        # Create FAISS index with documents and metadata
        global db
        db = create_faiss_index(docs, FAISS_DIR)

        return {"message": "File processed successfully.", "file": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.post("/query/")
async def query_model(question: str):
    """
    Endpoint to query the OpenAI model using the processed data.
    """
    try:
        if db is None:
            raise HTTPException(status_code=400, detail="No FAISS index found. Please upload a document first.")

        # Create QA chain
        qa_chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(),
            verbose=True
        )

        # Run query
        result = qa_chain({"question": question}, return_only_outputs=True)

        return {
            "answer": result.get("answer", "No answer found."),
            "sources": result.get("sources", "No sources available.")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying model: {str(e)}")
