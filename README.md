# PDF Chatbot RAG App

## About
A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that lets users upload PDF documents and interactively query their content. It:
- Splits and preprocesses PDF text with LangChain's `RecursiveCharacterTextSplitter`.
- Embeds chunks using HuggingFace embeddings (`all-MiniLM-L6-v2`) and indexes them with FAISS.
- Utilizes a history-aware retriever to reformulate follow-up queries using LangChain prompts.
- Generates concise, contextually relevant answers with Groq Gemma2 LLM.
- Maintains per-session chat history for conversational continuity.

## Installation
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

## Usage
1. Run the app:
   ```bash
   streamlit run project.py
   ```
2. Enter your **Groq API Key** and optional **Session ID**.
3. Upload one or more PDF files.
4. Ask questions in the input box; answers and chat history will display interactively.

## File Structure
```
.
├── project.py             # Main Streamlit application
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (Groq API Key)
```

## Dependencies
- streamlit
- langchain
- langchain-community
- langchain-core
- langchain-groq
- langchain-huggingface
- huggingface-hub
- langchain-text-splitters
- python-dotenv

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## License
MIT License © Rishi Chhabra

## Author
Built by Your Rishi Chhabra – [GitHub](https://github.com/rishichhabra29)
