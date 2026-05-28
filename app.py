import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import (
create_history_aware_retriever,
create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
create_stuff_documents_chain
)

# =========================

# LOAD ENV VARIABLES

# =========================

load_dotenv()

# =========================

# PAGE CONFIG

# =========================

st.set_page_config(
page_title="PDF RAG Chatbot",
page_icon="📄",
layout="wide"
)

st.title("📄 Conversational PDF RAG")
st.markdown("Upload PDF files and chat with your documents.")

# =========================

# SESSION STATE

# =========================

if "store" not in st.session_state:
st.session_state.store = {}

if "messages" not in st.session_state:
st.session_state.messages = []

# =========================

# API KEY

# =========================

groq_api_key = st.text_input(
"Enter GROQ API Key",
type="password"
)

if not groq_api_key:
st.warning("Please enter your GROQ API Key")
st.stop()

# =========================

# LLM

# =========================

llm = ChatGroq(
groq_api_key=groq_api_key,
model_name="llama-3.1-8b-instant",
temperature=0
)

# =========================

# EMBEDDINGS

# =========================

embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================

# SESSION ID

# =========================

session_id = st.text_input(
"Session ID",
value="default_session"
)

# =========================

# FILE UPLOADER

# =========================

uploaded_files = st.file_uploader(
"Upload PDF files",
type="pdf",
accept_multiple_files=True
)

# =========================

# PROCESS PDFs

# =========================

if uploaded_files:

```
documents = []

for uploaded_file in uploaded_files:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_pdf_path = temp_file.name

    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    documents.extend(docs)

# =========================
# TEXT SPLITTING
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(documents)

# =========================
# VECTOR STORE
# =========================
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# =========================
# CONTEXTUALIZE QUESTION
# =========================
contextualize_q_system_prompt = (
    "Given the chat history and latest user question, "
    "formulate a standalone question that can be understood "
    "without the chat history."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# =========================
# QA SYSTEM PROMPT
# =========================
system_prompt = (
    "You are an AI assistant for question-answering tasks. "
    "Use the retrieved context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Keep answers concise and accurate.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# =========================
# QA CHAIN
# =========================
question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)

# =========================
# SESSION HISTORY
# =========================
def get_session_history(session: str) -> BaseChatMessageHistory:

    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()

    return st.session_state.store[session]

# =========================
# CONVERSATIONAL RAG
# =========================
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# =========================
# CHAT INPUT
# =========================
user_input = st.chat_input(
    "Ask a question about your PDF..."
)

if user_input:

    # Store user message
    st.session_state.messages.append(
        ("user", user_input)
    )

    # Generate response
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    answer = response["answer"]

    # Store assistant response
    st.session_state.messages.append(
        ("assistant", answer)
    )

# =========================
# DISPLAY CHAT
# =========================
for role, message in st.session_state.messages:

    with st.chat_message(role):
        st.write(message)
```
