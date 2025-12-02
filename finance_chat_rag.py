import os
import shutil
from pathlib import Path
from typing import List
import stat
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


UPLOAD_DIR = "uploaded_pdfs"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "chroma_db_store"
LLM_MODEL = "llama3"
SIM_THRESHOLD = 0.6
TOP_K = 5


def ensure_upload_dir():
    Path(UPLOAD_DIR).mkdir(exist_ok=True)

def load_docs() -> List[Document]:
    loader = DirectoryLoader(UPLOAD_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks

def build_vectordb(chunks: List[Document]) -> Chroma:
    def on_rm_error(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            os.unlink(path)
        except Exception:
            pass

    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH, onerror=on_rm_error)
        except PermissionError:
            st.error("⚠️ Windows file lock detected! Please Stop the app, delete 'chroma_db_store', and restart.")
            return None

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    with st.spinner("🧠 Embedding documents... (This might take a moment)"):
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
        )
    return vectordb

class ThresholdRetriever(BaseRetriever):
    vectorstore: Chroma
    top_k: int = TOP_K
    sim_threshold: float = SIM_THRESHOLD

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None) -> List[Document]:
        results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        if not results: return []
        
        filtered = [doc for doc, score in results if score <= self.sim_threshold]
        
        if not filtered:
            filtered = [doc for doc, _ in results] # Fallback
            
        return filtered

def build_chain(vectordb: Chroma) -> RetrievalQA:
    retriever = ThresholdRetriever(
        vectorstore=vectordb,
        top_k=TOP_K,
        sim_threshold=SIM_THRESHOLD,
    )
    llm = OllamaLLM(model=LLM_MODEL)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return chain



def main():
    st.set_page_config(
        page_title="Finance RAG MVP",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

   
    st.markdown("""
    <style>
        .stChatMessage {border-radius: 10px; padding: 10px;}
        h1 {color: #2e7bcf;}
    </style>
    """, unsafe_allow_html=True)

    ensure_upload_dir()

 
    with st.sidebar:
        st.title("📈 Financial Analyst MVP")
        st.markdown("---")
        st.info("💡 **Prototype Mode**\nThis tool is the RAG engine for a larger autonomous trading system.")
        
        st.subheader("1. Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload IPO/Earnings Reports (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for f in uploaded_files:
                save_path = Path(UPLOAD_DIR) / f.name
                with open(save_path, "wb") as out_file:
                    out_file.write(f.getbuffer())
            st.caption(f"✅ {len(uploaded_files)} files staged.")

        if st.button("🔄 Build/Reset Vector DB", type="primary"):
            docs = load_docs()
            if not docs:
                st.warning("Please upload PDFs first.")
            else:
                chunks = chunk_docs(docs)
                vectordb = build_vectordb(chunks)
                if vectordb:
                    st.session_state["chain"] = build_chain(vectordb)
                    st.session_state["db_ready"] = True
                    st.success("Knowledge Base Updated!")
        
        st.markdown("---")
        st.caption(f"Core: {LLM_MODEL} | Embed: {EMBED_MODEL}")

  
    st.title("SEC Filing & IPO Researcher")
    st.markdown("Ask questions about the uploaded financial documents. The AI analyzes text chunks to provide evidence-based answers.")

    if "messages" not in st.session_state:
        st.session_state.messages = []


    if not st.session_state.get("db_ready"):
        st.warning("👈 Please upload documents and click **Build Vector DB** to start.")
        with st.chat_message("assistant"):
            st.write("I'm ready to analyze financial reports once you upload them.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("🔍 View Source Excerpts"):
                        st.markdown(message["sources"])

        if query := st.chat_input("Ex: What are the primary risk factors in the Meesho IPO?"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    chain = st.session_state["chain"]
                    result = chain({"query": query})
                    answer = result["result"]
                    source_docs = result["source_documents"]

                    source_text = ""
                    for i, doc in enumerate(source_docs):
                        source_name = Path(doc.metadata.get('source', 'unknown')).name
                        source_text += f"**Source {i+1} ({source_name}):**\n>{doc.page_content[:300]}...\n\n"

                message_placeholder.markdown(answer)
                with st.expander("🔍 View Source Excerpts"):
                    st.markdown(source_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_text
            })

if __name__ == "__main__":

    main()
