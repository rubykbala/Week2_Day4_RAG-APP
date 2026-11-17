# OpenAI chat model used to write answers from retrieved PDF text.
from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Streamlit builds the web page for our app.
import streamlit as st

# Loads settings and API keys from a local .env file when we add it later.
from dotenv import load_dotenv

# Read the .env file and make its values available as environment variables.
load_dotenv(override=True)

# Set the browser tab title and icon. This must run before other Streamlit commands.
st.set_page_config(page_title="Multi-PDF RAG App", page_icon="📚")

# Show the app title and a short instruction on the page.
st.title("Multi-PDF RAG App")
st.write("Upload PDF files and ask questions across them.")

# Let the user choose several PDF files in one upload.
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

# Tools for reading uploaded PDFs and splitting their text into chunks.
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Tools for temporarily saving uploaded files while PyPDFLoader reads them.
import os
import tempfile

def load_and_split_pdfs(uploaded_files):
    # Collect the pages from every uploaded PDF.
    all_pages = []

    for uploaded_file in uploaded_files:
        # PyPDFLoader reads from a file path, so save this upload temporarily.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            # Read the PDF one page at a time.
            pages = PyPDFLoader(temp_path).load()

            # Keep the original filename in each page's metadata.
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            all_pages.extend(pages)
        finally:
            # Remove the temporary copy after the PDF has been read.
            os.remove(temp_path)

    # Break the pages into smaller, overlapping pieces for better search.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(all_pages)

    return chunks

# Keep the processed chunks available while the app is running.
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Keep the processed chunks and search index available during this session.
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process the PDFs and build the shared search index when clicked.
if st.button("Process PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.warning("We’ll build the search index after adding your OpenAI API key.")
    else:
        with st.spinner("Reading PDFs and building the search index..."):
            chunks = load_and_split_pdfs(uploaded_files)

            # Turn each text chunk into an embedding for similarity search.
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

            # Build one FAISS index from chunks from all uploaded PDFs.
            st.session_state.vector_store = FAISS.from_documents(
                chunks,
                embeddings,
            )
            st.session_state.chunks = chunks

        st.success(f"Created {len(st.session_state.chunks)} text chunks and indexed them.")
        # Ask a question across all indexed PDFs.
st.subheader("Ask a question")
question = st.text_input("Enter your question about the uploaded PDFs")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif st.session_state.vector_store is None:
        st.warning("Upload PDFs and build the search index first.")
    else:
        # Find the four chunks most relevant to the question.
        matching_chunks = st.session_state.vector_store.similarity_search(
            question,
            k=4,
        )

        # Include each chunk's filename and page in the material given to the model.
        context_parts = []
        for chunk in matching_chunks:
            filename = chunk.metadata.get("source", "Unknown file")
            page_number = chunk.metadata.get("page", 0) + 1
            context_parts.append(
                f"[Source: {filename}, page {page_number}]\n{chunk.page_content}"
            )

        context = "\n\n".join(context_parts)

        # Ask the model to ground its answer in the retrieved PDF text.
        prompt = f"""
You answer questions using only the provided PDF excerpts.
If excerpts disagree, explain the conflicting information and identify
which source gives each version. If the excerpts do not answer the
question, say so clearly. Do not invent facts or citations.

Question:
{question}

PDF excerpts:
{context}

Give a concise answer and cite supporting excerpts by filename and page.
"""
        with st.spinner("Searching the PDFs and writing an answer..."):
            chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            answer = chat_model.invoke(prompt)

        st.markdown("### Answer")
        st.write(answer.content)

        # Show the retrieved source locations beneath the answer.
        st.markdown("### Retrieved sources")
        for chunk in matching_chunks:
            filename = chunk.metadata.get("source", "Unknown file")
            page_number = chunk.metadata.get("page", 0) + 1
            st.write(f"- {filename}, page {page_number}")
