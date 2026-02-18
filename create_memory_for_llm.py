from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ===== STEP 1: LOAD PDFs =====
DATA_PATH = r"C:\Users\swale\OneDrive\Desktop\medical-chatbot-main\data"

def load_pdf_files(path):
    loader = DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

documents = load_pdf_files(DATA_PATH)
print(f"PDF Pages Loaded: {len(documents)}")

# ===== STEP 2: SPLIT INTO CHUNKS =====
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

chunks = split_docs(documents)
print(f"Text Chunks Created: {len(chunks)}")

# ===== STEP 3: EMBEDDINGS =====
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===== STEP 4: STORE IN FAISS =====
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print("✅ FAISS vectorstore created successfully")
