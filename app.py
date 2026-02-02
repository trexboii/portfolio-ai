import streamlit as st
from google import genai
from google.genai import types
import os
import time
import requests

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
@st.cache_resource
@st.cache_resource
def index_portfolios():
    """Automatically discovers and downloads PDFs from the latest GitHub Release."""
    # CHANGE THESE to your actual details
    GITHUB_OWNER = "trexboii"
    GITHUB_REPO = "portfolio-ai"
    
    # GitHub API endpoint for the latest release
    api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
    
    uploaded_uris = []
    file_names = []
    
    try:
        # 1. Ask GitHub for the list of files (assets) in the release
        response = requests.get(api_url)
        response.raise_for_status()
        assets = response.json().get("assets", [])
        
        if not assets:
            st.warning("No files found in the GitHub Release!")
            return [], []

        # 2. Loop through every file found
        for asset in assets:
            filename = asset["name"]
            
            # Only process PDF portfolios
            if filename.lower().endswith(".pdf"):
                download_url = asset["browser_download_url"]
                
                # Download the file to a temp location
                file_data = requests.get(download_url).content
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name
                
                # 3. Upload to Gemini
                file_ref = client.files.upload(file=tmp_path)
                while file_ref.state == "PROCESSING":
                    time.sleep(1)
                    file_ref = client.files.get(name=file_ref.name)
                
                uploaded_uris.append(file_ref)
                file_names.append(filename)
                os.remove(tmp_path)
                
        return uploaded_uris, file_names

    except Exception as e:
        st.error(f"Failed to sync with GitHub: {e}")
        return [], []

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