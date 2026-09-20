import os
import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

st.title("Legal Advisory Assistant (Powered by Groq)")

# Read key from Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.warning("Please add your GROQ_API_KEY to your secrets.toml file inside the .streamlit folder.")
    st.stop()

# Helper function to format documents into text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# PURE PYTHON TEXT SPLITTER
def split_text_manually(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    if not text.strip():
        return chunks
    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end]
        chunks.append(Document(page_content=chunk_content))
        start += (chunk_size - chunk_overlap)
    return chunks

# Keyword Search Engine 
def keyword_retriever(query, documents, k=3):
    keywords = query.lower().split()
    scored_docs = []
    for doc in documents:
        score = sum(1 for word in keywords if word in doc.page_content.lower())
        scored_docs.append((score, doc))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:k]]

# AUTOMATIC FOLDER DETECTOR
pdf_folder = "docs"
if os.path.exists("docs/doc") and any(f.endswith('.pdf') for f in os.listdir("docs/doc")):
    pdf_folder = "docs/doc"
elif os.path.exists("docs") and any(f.endswith('.pdf') for f in os.listdir("docs")):
    pdf_folder = "docs"

# Verify folder status
if not os.path.exists(pdf_folder):
    st.warning("Please create a 'docs' folder and put your PDF inside it.")
    st.stop()

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

if not pdf_files:
    st.warning(f"No PDFs found. Please drop your legal files into your folder.")
    st.stop()

# CACHE TEXT EXTRACTION
@st.cache_resource
def load_and_process_pdfs(folder, files):
    all_pdf_text = ""
    for filename in files:
        pdf_path = os.path.join(folder, filename)
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_pdf_text += text + "\n"
        except Exception:
            pass
    return split_text_manually(all_pdf_text)

split_docs = load_and_process_pdfs(pdf_folder, pdf_files)

if not split_docs:
    st.error("Could not extract readable text from your PDF files. Ensure they are text-based PDFs.")
    st.stop()
else:
    st.sidebar.success(f"Loaded {len(pdf_files)} PDF(s) into {len(split_docs)} text fragments!")

# Set up the Groq AI Model
llm = ChatGroq(temperature=0, model_name="qwen/qwen3.8-27b")

# Define System Instructions
system_prompt = (
    "You are an expert legal advisory chatbot. Answer the user's question "
    "clearly and strictly using the provided legal context below. If you "
    "do not know the answer based on the context, say that you don't know.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

rag_chain = prompt | llm | StrOutputParser()

# Track and show messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle chat box input
user_query = st.chat_input("Ask a legal question:")
if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    # Fetch matching context via keyword matcher
    matched_docs = keyword_retriever(user_query, split_docs, k=3)
    context_text = format_docs(matched_docs)

    formatted_history = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            formatted_history.append(("human", msg["content"]))
        else:
            formatted_history.append(("ai", msg["content"]))

    with st.chat_message("assistant"):
        response = rag_chain.invoke({
            "context": context_text,
            "input": user_query,
            "chat_history": formatted_history
        })
        st.write(response)

    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
