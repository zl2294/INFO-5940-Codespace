import os
import streamlit as st
from typing import List

from os import environ
if not environ.get("OPENAI_API_KEY"):
    environ["OPENAI_API_KEY"] = os.environ.get("API_KEY", "")
if not environ.get("OPENAI_BASE_URL"):
    environ["OPENAI_BASE_URL"] = "https://api.ai.it.cornell.edu"

# LangChain model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# set up the Streamlit UI
st.set_page_config(page_title="RAG Chat", page_icon="💬")
st.title("📝 File Q&A with OpenAI")

PERSIST_DIR = "./chroma_db"
COLLECTION = "docs"

def get_vectorstore() -> Chroma:
    if "vs" not in st.session_state:
        embeddings = OpenAIEmbeddings(
            model="openai.text-embedding-3-large",               
            openai_api_key=environ["OPENAI_API_KEY"],       
            openai_api_base=environ["OPENAI_BASE_URL"],          
        )
        st.session_state["vs"] = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    return st.session_state["vs"]

# read txt file 
def read_txt(f) -> Document:
    text = f.read().decode("utf-8", errors="ignore")
    return Document(page_content=text, metadata={"source": f.name})
# read pdf file 
def read_pdf(f) -> List[Document]:
    try:
        from pypdf import PdfReader
    except Exception:
        from PyPDF2 import PdfReader
    reader = PdfReader(f)
    out: List[Document] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        out.append(Document(page_content=text, metadata={"source": f.name, "page": i + 1}))
    return out

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

# can upload multiple file including txt and pdf 
uploaded_files = st.file_uploader(
    "Upload documents with .txt or/and .pdf file",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    raw_docs: List[Document] = []
    for f in uploaded_files:
        name = f.name.lower()
        if name.endswith(".txt"):
            raw_docs.append(read_txt(f))
        else:
            raw_docs.extend(read_pdf(f))

    chunks = split_docs(raw_docs)
    vs = get_vectorstore()
    if chunks:
        vs.add_documents(chunks) 
    st.success(f"Indexed {len(uploaded_files)} files, {len(chunks)} chunks")

# remember the history 
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "I will answer the question based on your documents."}
    ]

for m in st.session_state["messages"]:
    st.chat_message(m["role"]).write(m["content"])

# LLM 
llm = ChatOpenAI(
    model="openai.gpt-4o",                     
    temperature=0.2,
    openai_api_key=environ["OPENAI_API_KEY"],
    openai_api_base=environ["OPENAI_BASE_URL"],
)

# ask question, retrievel and generated 
def build_context(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

def lc_history_from_streamlit(history: List[dict]) -> List:
    out = []
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out

user_q = st.chat_input("How can I help you?")
if user_q:
    st.session_state["messages"].append({"role": "user", "content": user_q})
    st.chat_message("user").write(user_q)

    with st.chat_message("assistant"):
        try:
            # retrivel 
            vs = get_vectorstore()
            # cosine similarity search via vector retriever
            retrieved = vs.as_retriever(search_kwargs={"k": 4}).invoke(user_q)
            if not retrieved:
                st.write("I don't know based on the provided documents.")
                st.session_state["messages"].append({"role": "assistant", "content": "I don't know based on the provided documents."})
                st.stop()
            context = build_context(retrieved)
            sys = SystemMessage(
                content="You are a helpful assistant. Answer ONLY using the provided context. "
                        "If the answer is not in the context, say you don't know. Be concise."
            )
            usr = HumanMessage(content=f"Question: {user_q}\n\nContext:\n{context}")

            messages = [sys, usr]

            # generate 
            result = llm.invoke(messages)
            answer = result.content

            # cited the source 
            if retrieved:
                cites = []
                for d in retrieved:
                    src = d.metadata.get("source", "")
                    page = d.metadata.get("page")
                    cites.append(f"{src}" + (f":p{page}" if page else ""))
                cites = "; ".join(dict.fromkeys(cites))
                answer += f"\n\nSources: {cites}"
            st.write(answer)
        except Exception as e:
            answer = f"Error: {e}"
            st.error(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
