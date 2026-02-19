import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

PDF_DIR = "pdfs"
INDEX_DIR = "faiss_index"

def extract_all_pdfs(pdf_dir):
    text = ""
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_dir, filename))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    return text

def main():
    raw_text = extract_all_pdfs(PDF_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(raw_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)

    vector_store.save_local(INDEX_DIR)
    print("✅ FAISS index built and saved")

if __name__ == "__main__":
    main()
