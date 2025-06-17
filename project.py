import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG with PDF uploads and document history")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your Groq API Key:", type="password")
if not api_key:
    st.warning("Please enter the Groq API key")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
session_id = st.text_input("Session ID", value="default_session")


if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)
if uploaded_files:

    docs = []
    for f in uploaded_files:
        with open("temp.pdf", "wb") as tmp:
            tmp.write(f.getbuffer())
        docs.extend(PyPDFLoader("temp.pdf").load())
    splits = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500).split_documents(docs)

    # index
    vector_store = FAISS.from_documents(splits, embeddings)
    retriever = vector_store.as_retriever()

    # turn history + query → standalone q
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Given the chat history and the latest user question—which might reference "
         "context—reformulate it into a self-contained question or return it as-is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an assistant for question-answering tasks. Use the retrieved context to answer "
         "the question in 2 paragragphs. If you don't know, say so. Keep answers under three sentences.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversation = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",   
        output_messages_key="answer"
    )

    
    question = st.text_input("Your Question:")
    if question:
        out = conversation.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        st.write("Assistant:", out["answer"])
        st.write("---")
        st.write("History:", get_session_history(session_id).messages)
