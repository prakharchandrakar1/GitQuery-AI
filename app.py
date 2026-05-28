import streamlit as st
import shutil
import os
import git

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

load_dotenv()

SUPPORTED_EXTENSIONS = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.md', '.txt']

def clone_repo(repo_url: str, clone_path: str = "./cloned_repo"):
    if os.path.exists(clone_path):
        return clone_path
    git.Repo.clone_from(repo_url, clone_path)
    return clone_path

def load_code_files(repo_path: str):
    docs = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in SUPPORTED_EXTENSIONS:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    relative_path = os.path.relpath(filepath, repo_path)
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": relative_path}
                    ))
                except Exception:
                    continue
    return docs

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def build_qa_chain(retriever):
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain


st.set_page_config(page_title="GitHub Q&A Bot", page_icon="🤖")
st.title("GitHub Repository Q&A Bot")

repo_url = st.text_input("Paste a GitHub repo URL", placeholder="https://github.com/user/repo")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if st.button("Ingest Repository") and repo_url:
    with st.spinner("Cloning and processing repo..."):
        shutil.rmtree("./cloned_repo", ignore_errors=True)
        shutil.rmtree("./chroma_db", ignore_errors=True)

        path = clone_repo(repo_url)
        docs = load_code_files(path)
        st.info(f"Loaded {len(docs)} files")

        vectorstore = build_vectorstore(docs)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        st.session_state.qa_chain = build_qa_chain(retriever)
        st.success("Repository ingested! Ask your questions below.")

if st.session_state.qa_chain:
    question = st.chat_input("Ask something about this codebase...")
    if question:
        result = st.session_state.qa_chain({
            "question": question,
            "chat_history": st.session_state.chat_history
        })
        answer = result["answer"]
        sources = list(set([
            doc.metadata["source"] for doc in result["source_documents"]
        ]))

        st.session_state.chat_history.append((question, answer))

        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
        st.caption("Sources: " + ", ".join(sources))
        
        