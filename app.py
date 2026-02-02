import streamlit as st
from google import genai
from google.genai import types
import os
import time

# --- 1. SETUP & SECRETS ---
st.set_page_config(page_title="Portfolio Expert AI", page_icon="🧠")

# Get API Key from Streamlit Secrets (Secure way)
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("API Key not found. Please set it in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# --- 2. CACHING SYSTEM ---
# We use @st.cache_resource so we don't re-upload files every time 
# the user clicks a button. It only runs once per session.
@st.cache_resource
def index_portfolios():
    """Reads files from the 'portfolios' folder and uploads them to Gemini."""
    portfolio_folder = "portfolios"
    uploaded_uris = []
    file_names = []

    # Check if folder exists
    if not os.path.exists(portfolio_folder):
        return [], ["Error: 'portfolios' folder not found!"]

    files = [f for f in os.listdir(portfolio_folder) if f.endswith('.pdf')]
    
    if not files:
        return [], ["No PDF files found in the folder."]

    # Upload files
    for filename in files:
        file_path = os.path.join(portfolio_folder, filename)
        
        # Upload to Gemini
        file_ref = client.files.upload(file=file_path)
        
        # Wait for processing
        while file_ref.state == "PROCESSING":
            time.sleep(1)
            file_ref = client.files.get(name=file_ref.name)
            
        uploaded_uris.append(file_ref)
        file_names.append(filename)
        
    return uploaded_uris, file_names

# --- 3. UI LAYOUT ---
st.title("🤖 Portfolio Insight Bot")
st.markdown("I have studied **all the team portfolios**. Ask me anything!")

# Load the brain (this runs silently in the background)
with st.spinner("Waking up the AI..."):
    stored_files, file_names = index_portfolios()

# Show users what data is loaded (optional, good for trust)
with st.expander(f"📚 Knowledge Base ({len(file_names)} files loaded)"):
    for name in file_names:
        st.write(f"- {name}")

# --- 4. CHAT LOGIC ---
if "chat_history" not in st.session_state:
    # Give the bot a personality in the first message
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I've analyzed the portfolios. Ask me to compare teams, find unique ideas, or spot trends."}
    ]

# Display Chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle Input
if user_input := st.chat_input("Ex: 'Which team had the best user research?'"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Combine files + user question
                prompt_content = stored_files + [user_input]
                
                response = client.models.generate_content(
                    model="gemini-2.0-flash", # Using the stable Flash model
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful expert. Answer questions based ONLY on the uploaded portfolio files. If the answer isn't in the files, say so."
                    )
                )
                
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error: {e}")