import os
import shutil
from langchain.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

def create_faiss_index(docs, faiss_dir):
    """
    Creates a FAISS index from the given documents.
    """
    # Remove old FAISS index
    if os.path.exists(faiss_dir):
        shutil.rmtree(faiss_dir)

    # Create and save the new index
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    db.save_local(faiss_dir)
    return db

'''def load_faiss_index(faiss_dir):
    """
    Loads an existing FAISS index.
    """
    return FAISS.load_local(faiss_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)'''
