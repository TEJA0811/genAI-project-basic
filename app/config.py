import os

# Directories
UPLOAD_DIR = "uploaded_files"
FAISS_DIR = "../faiss_index"

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# Logging configuration
LOG_FILE = "logs/langchain_debug.log"
LOG_DIR = os.path.dirname(LOG_FILE)
os.makedirs(LOG_DIR, exist_ok=True)
