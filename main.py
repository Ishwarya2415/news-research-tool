import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
if not gemini_api_key:
    st.error("❗ GEMINI_API_KEY not found. Please check your .env file!")
    st.stop()

genai.configure(api_key=gemini_api_key)

# Streamlit UI setup
st.title("🚀 RockyBot: News Research Tool 📈")
st.sidebar.title("🔍 Enter News Article URLs")

# Collect URLs from sidebar
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]

process_url_clicked = st.sidebar.button("✨ Process URLs")
file_path = "faiss_store_gemini.pkl"
main_placeholder = st.empty()

# Function to fetch content from URLs
def fetch_content_fallback(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text(separator="\n")

# Step 1: Load content
if process_url_clicked:
    data = [fetch_content_fallback(url) for url in urls if url]

    if not data:
        st.error("❗ Failed to fetch content from all provided URLs. Please check the links!")
        st.stop()

    st.success(f"✅ Loaded content from {len(data)} URL(s)!")

    # Step 2: Split data into smaller chunks with source metadata
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    docs = []
    for content, url in zip(data, urls):
        split_chunks = text_splitter.split_text(content)
        for chunk in split_chunks:
            docs.append({"page_content": chunk, "metadata": {"source": url}})

    if not docs:
        st.error("❗ No content chunks found. The articles might be too short or unreadable.")
        st.stop()

    st.success(f"✅ Split content into {len(docs)} chunks!")

    # Step 3: Create embeddings with SentenceTransformer
    from langchain.embeddings import HuggingFaceEmbeddings

    # Load the embeddings model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vector store correctly with text + metadata
    vectorstore_gemini = FAISS.from_texts(
        texts=[doc["page_content"] for doc in docs],
        embedding=embedding_model,
        metadatas=[doc["metadata"] for doc in docs]
    )

    st.success("✅ Embedding Vector Built Successfully!")
    time.sleep(2)

    # Save FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_gemini, f)

# Step 4: Accept user query
query = main_placeholder.text_input("🔎 Ask a question based on the articles:")

# Step 5: Process the query
if query:
    if os.path.exists(file_path):
        # Load FAISS index
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Find the most relevant content chunk
        query_embedding = embedding_model.embed_query(query)
        docs_and_scores = vectorstore.similarity_search_by_vector(query_embedding, k=3)

        # Extract content and the corresponding source URLs
        if docs_and_scores:
            context = "\n\n".join([doc.page_content for doc in docs_and_scores])
            sources = list(
                set(doc.metadata.get('source', '🔗 Source not available') for doc in docs_and_scores)
            )  # Use set() to remove duplicates

            # Generate answer with Gemini model
            gemini_model = genai.GenerativeModel("gemini-1.5-pro")
            response = gemini_model.generate_content(f"Based on this context: {context}, answer: {query}")

            # Display the answer
            st.header("✅ Answer:")
            st.write(response.text if hasattr(response, 'text') else response)

            # Show the unique source URLs that contributed to the answer
            st.subheader("📌 Sources (most relevant):")
            for source in sources:
                st.write(f"🔗 [{source}]({source})")

        else:
            st.warning("❗ No relevant content found for this query.")
    else:
        st.error("❗ FAISS index file not found. Please process URLs first!")
