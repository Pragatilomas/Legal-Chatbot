import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load API Key from your .env file
load_dotenv()

st.set_page_config(page_title="Legal Chatbot", page_icon="⚖️")
st.title("⚖️ Legal Advisory Assistant")

# Initialize the free Groq Client
if "groq_client" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.session_state.groq_client = Groq(api_key=api_key)
    else:
        st.session_state.groq_client = None

# Check if legal data exists
DATA_FILE = "legal_data.txt"

if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
    
    # Simple search function to find relevant paragraphs without needing PyTorch
    def find_relevant_context(query, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            paragraphs = f.read().split("\n\n") # Splitting data by double lines
        
        # Look for matching words
        query_words = set(query.lower().split())
        scored_paragraphs = []
        
        for para in paragraphs:
            if not para.strip():
                continue
            # Count how many words match the user query
            matches = sum(1 for word in query_words if word in para.lower())
            if matches > 0:
                scored_paragraphs.append((matches, para))
        
        # Sort by best match and take top 3
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])
        top_matches = [para for score, para in scored_paragraphs[:3]]
        
        return "\n\n".join(top_matches) if top_matches else "No specific context found."

    # Chat history state setup
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User interacts
    if user_query := st.chat_input("Ask a legal question..."):
        if not st.session_state.groq_client:
            st.error("❌ Groq API Key missing! Please add GROQ_API_KEY=your_key inside your hidden .env file.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Reviewing legal documents..."):
                try:
                    # Fetch matching text from legal_data.txt
                    context = find_relevant_context(user_query, DATA_FILE)
                    
                    # Package instructions for the AI model
                    system_instructions = (
                        f"You are a helpful legal advisory chatbot. "
                        f"Answer the question using this reference context from the user's files:\n\n{context}\n\n"
                        f"If the answer isn't in the context, give a general helpful legal response."
                    )
                    
                    messages_payload = [{"role": "system", "content": system_instructions}]
                    for msg in st.session_state.messages[-6:]: # Keep memory of last few messages
                        messages_payload.append({"role": msg["role"], "content": msg["content"]})
                    
                    # Request generation from Groq (Llama 3.3 model is free and fast)
                    completion = st.session_state.groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages_payload,
                        temperature=0.2,
                    )
                    
                    response_text = completion.choices[0].message.content
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.warning("⚠️ Welcome! Please open your 'legal_data.txt' file in VS Code and paste some legal references, sections, or documentation inside it to begin.")
