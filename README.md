# ⚖️ AI Legal Advisory Chatbot (RAG Engine)

A privacy-focused, zero-local-dependency **Retrieval-Augmented Generation (RAG)** legal assistant. The application reads context directly from text-based legal PDF acts (such as the *Constitution of India*) and leverages cloud-computed word vectors and advanced reasoning models to deliver grounded legal insights without heavy local compilation requirements.

## 🚀 Core Features
*   **Pure Cloud Processing**: Bypasses heavy local deep learning libraries (`torch`, `sentence-transformers`), making it highly stable on lightweight operating systems or experimental runtimes.
*   **Intelligent Keyword Search Engine**: Scans complex multi-page PDF documents dynamically to find matching paragraphs based on real language relevance structures.
*   **Fully Contextual Chat Memory**: Tracks ongoing conversations to effortlessly understand split context and conversational pronouns during follow-up user questions.
*   **Secure Credential Architecture**: Built native to Streamlit secrets configurations to isolate production credentials and guarantee runtime protection.

## 🛠️ Tech Stack & Components
*   **UI Framework**: Streamlit (Native Chat Interface elements)
*   **AI Inference Engine**: Groq Cloud Services (`qwen/qwen3.8-27b` Reasoning Architecture)
*   **Orchestration**: LangChain Framework Core
*   **PDF Extraction**: PyPDF Engine Calculations

## 📦 Local Installation Guide

1. Clone this repository to your workspace:
   ```bash
   git clone https://github.com/Pragatilomas/Legal-Chatbot.git
   cd Legal-Chatbot
   ```

2. Install the required lightweight processing libraries:
   ```bash
   pip install streamlit pypdf langchain-community langchain-core langchain-groq langchain-openai
   ```

3. Create a `.streamlit/secrets.toml` file in the root workspace directory and add your free developer token:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```

4. Drop your target legal acts (in text-selectable PDF format) directly into the `docs/doc` structural directory pathway.

5. Boot up the local runtime compilation environment:
   ```bash
   streamlit run app.py
   ```
