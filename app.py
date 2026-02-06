import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tempfile

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="FTC Robotics Resource Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for FTC branding
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. API KEY SETUP ---
def setup_api_key():
    """Load API key from Streamlit secrets or .env file"""
    api_key = None
    
    # Try Streamlit secrets first (for deployment)
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fall back to .env file (for local development)
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("❌ **Google API Key not found!** Please add it to `.env` or Streamlit Secrets.")
        st.info("""
        **To get started:**
        1. Get a Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create a `.env` file in your project directory
        3. Add this line: `GOOGLE_API_KEY=your-api-key-here`
        """)
        st.stop()
    
    os.environ["GOOGLE_API_KEY"] = api_key
    return api_key

# --- 3. BACKEND FUNCTIONS ---

def extract_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    if not pdf_docs:
        return ""
    
    text = ""
    for pdf_file in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {str(e)}")
    
    return text

def create_text_chunks(text):
    """Split text into manageable chunks for embedding"""
    if not text or len(text.strip()) == 0:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    """Create FAISS vector store from text chunks"""
    if not text_chunks:
        return None
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )
        vector_store = FAISS.from_texts(
            texts=text_chunks, 
            embedding=embeddings
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_tokens=10000
    )

    system_prompt = """You are a helpful assistant for FIRST Tech Challenge (FTC) robotics teams.
Answer questions using ONLY the provided context.
If the answer is not contained in the documents, say you do not have enough information.
Do NOT make up rules or interpretations.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    rag_chain = (
        {
            "context": retriever | format_docs,  # 🔑 FIX
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def process_question(chain, question):
    try:
        return chain.invoke(question)
    except Exception as e:
        return f"Error processing question: {str(e)}"


# --- 4. SESSION STATE INITIALIZATION ---
def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "👋 Welcome to the FTC Robotics Resource Chatbot! Upload your PDF documents (game manuals, rules, guides) in the sidebar and click 'Process Documents' to get started."
            }
        ]
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

def format_docs(docs):
    return "\n\n".join(
        f"[Source]\n{doc.page_content}" for doc in docs
    )


# --- 5. MAIN APPLICATION ---
def main():
    # Setup
    setup_api_key()
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🤖 FTC Robotics Resource Chatbot</p>', unsafe_allow_html=True)
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📚 Document Management")
        
        st.markdown("""
        Upload FTC-related PDFs:
        - Game Manuals
        - Rule Books
        - Programming Guides
        - Team Resources
        """)
        
        pdf_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to create your knowledge base"
        )
        
        if st.button("🔄 Process Documents", type="primary", use_container_width=True):
            if not pdf_files:
                st.warning("⚠️ Please upload at least one PDF file first.")
            else:
                with st.spinner("Processing documents... This may take a moment."):
                    # Extract text
                    raw_text = extract_pdf_text(pdf_files)
                    
                    if not raw_text:
                        st.error("No text could be extracted from the PDFs. Please check your files.")
                        return
                    
                    # Create chunks
                    text_chunks = create_text_chunks(raw_text)
                    
                    if not text_chunks:
                        st.error("Could not create text chunks. Please check your documents.")
                        return
                    
                    st.info(f"Created {len(text_chunks)} text chunks from your documents.")
                    
                    # Create vector store
                    vector_store = create_vector_store(text_chunks)
                    
                    if vector_store is None:
                        st.error("Failed to create vector store. Please try again.")
                        return
                    
                    # Create RAG chain
                    rag_chain = get_conversational_chain(vector_store)
                    
                    # Store in session state
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_chain = rag_chain
                    st.session_state.processed_files = [f.name for f in pdf_files]
                    
                    st.success(f"✅ Successfully processed {len(pdf_files)} document(s)!")
        
        # Show processed files
        if st.session_state.processed_files:
            st.divider()
            st.subheader("📄 Loaded Documents")
            for filename in st.session_state.processed_files:
                st.text(f"• {filename}")
        
        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared! How can I help you with FTC robotics?"
                }
            ]
            st.rerun()
    
    # Main chat interface
    st.subheader("💬 Ask Questions About Your Documents")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_question := st.chat_input("Ask a question about FTC robotics..."):
        
        # Check if documents are processed
        if st.session_state.rag_chain is None:
            st.error("⚠️ Please upload and process documents first using the sidebar!")
            return
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_question(st.session_state.rag_chain, user_question)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer info
    with st.expander("ℹ️ About this chatbot"):
        st.markdown("""
        This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions about 
        FIRST Tech Challenge (FTC) robotics based on the documents you upload.
        
        **How it works:**
        1. Upload FTC-related PDF documents
        2. The system processes and indexes the content
        3. Ask questions and get answers based on your documents
        
        **Tips for best results:**
        - Upload official FTC game manuals and rule books
        - Ask specific questions about rules, scoring, or technical requirements
        - Reference specific game elements or scenarios
        
        **Technology Stack:**
        - Streamlit for the interface
        - Google Gemini AI for language understanding
        - LangChain for RAG implementation
        - FAISS for vector storage
        """)

if __name__ == "__main__":
    main()