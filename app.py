import streamlit as st
import os
#i am the keeper of time
import time
from pypdf import PdfReader
from dotenv import load_dotenv


from langchain_core.runnables import RunnablePassthrough, RunnableLambda # Add RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# CONFIG
# =========================================================

INDEX_DIR = "faiss_index"

st.set_page_config(
    page_title="FTC Robotics Resource Chatbot",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# API KEY
# =========================================================

def setup_api_key():
    load_dotenv()

    api_key = None
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("❌ GOOGLE_API_KEY not found.")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = api_key


# =========================================================
# PDF UTILITIES
# =========================================================

def extract_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


# =========================================================
# VECTOR STORES
# =========================================================

def load_base_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


def build_session_vector_store(pdf_files):
    raw_text = extract_pdf_text(pdf_files)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(raw_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    # --- BATCH PROCESSING FIX ---
    vector_store = None
    batch_size = 10  # Process 10 chunks at a time
    
    # Optional: Show a progress bar in Streamlit
    progress_text = "Embedding documents..."
    my_bar = st.progress(0, text=progress_text)

    for i in range(0, len(chunks), batch_size):
        # Slice the chunks for the current batch
        batch_chunks = chunks[i : i + batch_size]
        
        if vector_store is None:
            # First batch creates the vector store
            vector_store = FAISS.from_texts(batch_chunks, embeddings)
        else:
            # Subsequent batches add to the existing store
            vector_store.add_texts(batch_chunks)
        
        # Update progress bar
        progress_percent = min((i + batch_size) / len(chunks), 1.0)
        my_bar.progress(progress_percent, text=f"{progress_text} ({int(progress_percent*100)}%)")

        # IMPORTANT: Sleep to respect rate limits and avoid server disconnects
        time.sleep(1)

    my_bar.empty() # Clear the progress bar when done
    return vector_store

# =========================================================
# RAG CHAIN (BASE + SESSION)
# =========================================================

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def get_combined_retriever(base_vs, session_vs):
    base_retriever = base_vs.as_retriever(search_kwargs={"k": 3})

    if session_vs:
        session_retriever = session_vs.as_retriever(search_kwargs={"k": 3})
    else:
        session_retriever = None

    def retrieve(query):
        docs = []
        # invoke() is the standard way to call retrievers in modern LangChain
        docs.extend(base_retriever.invoke(query))

        if session_retriever:
            docs.extend(session_retriever.invoke(query))

        return docs

    # WRAP THE FUNCTION HERE so it supports the | operator
    return RunnableLambda(retrieve)


def get_rag_chain(base_vs, session_vs):
    retriever = get_combined_retriever(base_vs, session_vs)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        timeout=None,
        max_retries=2
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant for FIRST Tech Challenge (FTC) robotics. "
            "Answer ONLY using the provided context."
            "When someone asks for ideas, provide examples based on the context and then give a summary of key information"
            "If the answer is not contained in the documents, say you do not know."
            "Do not cut off in the middle of sentences."
            "Do not use the names of specific people from teams."
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ])

    return (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# =========================================================
# SESSION STATE
# =========================================================

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "base_vector_store" not in st.session_state:
        st.session_state.base_vector_store = None

    if "session_vector_store" not in st.session_state:
        st.session_state.session_vector_store = None

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None


# =========================================================
# MAIN APP
# =========================================================

def main():
    setup_api_key()
    init_session_state()

    st.title("🤖 FTC Robotics Resource Chatbot")

    # -------- Load base index once --------
    if st.session_state.base_vector_store is None:
        if not os.path.exists(INDEX_DIR):
            st.error("Base FAISS index not found.")
            st.stop()

        st.session_state.base_vector_store = load_base_vector_store()
        st.session_state.rag_chain = get_rag_chain(
            st.session_state.base_vector_store,
            None
        )

    # -------- Sidebar --------
    with st.sidebar:
        st.header("📚 Your Session Documents")

        pdf_files = st.file_uploader(
            "Upload PDFs (session-only)",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("➕ Add to My Session"):
            if not pdf_files:
                st.warning("Upload at least one PDF.")
            else:
                with st.spinner("Adding documents to your session..."):
                    st.session_state.session_vector_store = (
                        build_session_vector_store(pdf_files)
                    )
                    st.session_state.rag_chain = get_rag_chain(
                        st.session_state.base_vector_store,
                        st.session_state.session_vector_store
                    )
                st.success("Documents added for this session only.")

        if st.session_state.session_vector_store:
            st.divider()
            if st.button("🧹 Clear Session Documents"):
                st.session_state.session_vector_store = None
                st.session_state.rag_chain = get_rag_chain(
                    st.session_state.base_vector_store,
                    None
                )
                st.success("Session documents cleared.")
                st.rerun()

    # -------- Chat UI --------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about FTC robotics..."):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(question)
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()
