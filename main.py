import os
import logging
import streamlit as st
import time
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import chromadb
from dotenv import load_dotenv
import langchain
import shutil
import re
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import PromptTemplate
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma

# Enable debugging
langchain.debug = True

# Configure logging to Streamlit
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("langchain_debug")
file_handler = logging.FileHandler("langchain_debug.log")
logger.addHandler(file_handler)

file_path = "C:\\Users\\tejas\\Desktop\\myDoc\\Major project-report.pdf"

load_dotenv()  # Load environment variables

st.title("Unlock the wisdom in your data.")
st.sidebar.title("File Upload")

# Initialize session state for graph, db, and chain
if "graph" not in st.session_state:
    st.session_state.graph = nx.Graph()
if "db" not in st.session_state:
    st.session_state.db = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "query" not in st.session_state:
    st.session_state.query = ""
if "linked_docs" not in st.session_state:
    st.session_state.linked_docs = None
if "refined_query" not in st.session_state:
    st.session_state.refined_query = ""
if "docs" not in st.session_state:
    st.session_state.docs = None

process_file_clicked = st.sidebar.button("Upload")

main_placeholder = st.empty()
llm = ChatOpenAI(model="gpt-4o-mini")

# Function for cleaning raw text
def clean_text(raw_text):
    """
    Cleans extracted text to remove headers, footers, and line breaks mid-sentence.
    """
    cleaned = re.sub(r'\s+', ' ', raw_text)
    cleaned = re.sub(r'Page \d+ of \d+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(\w)-\n(\w)', r'\1\2', cleaned)
    cleaned = re.sub(r'\n', ' ', cleaned)
    return cleaned

# Function to build a graph of documents
def build_document_graph(input_docs, graph):
    embedder = OpenAIEmbeddings()
    embeddings = [embedder.embed_query(doc.page_content) for doc in input_docs]
    for i, emb1 in enumerate(embeddings):
        for j, emb2 in enumerate(embeddings):
            if i != j:
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                if similarity > 0.8:  # Threshold for linking
                    graph.add_edge(i, j, weight=similarity)

# Function to retrieve linked documents from the graph
def retrieve_linked_documents(graph, input_refined_query, input_docs):
    embedder = OpenAIEmbeddings()

    # Ensure input_refined_query is a string
    query_text = input_refined_query if isinstance(input_refined_query, str) else input_refined_query.content

    query_embedding = embedder.embed_query(query_text)

    results = []
    for node in graph.nodes(data=True):
        content = input_docs[node[0]].page_content
        node_embedding = embedder.embed_query(content)
        similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
        logger.debug(f"Similarity between query and document {node[0]}: {similarity}")

        # Lower the similarity threshold (from 0.8 to 0.6)
        if similarity > 0.6:  # Adjusted threshold
            results.append((node[0], similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    logger.debug(f"Found {len(results)} documents with similarity > 0.6")
    return [input_docs[node_id] for node_id, _ in results]

# Query Rewriter Agent
def rewrite_query(query):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Rephrase this query to make it more specific and focused for document retrieval: {query}. Provide a direct, concise refinement without additional suggestions."
    )
    return llm.invoke(prompt.format(query=query))

# Answer Generator Agent using Chroma
def generate_answer(query, input_linked_docs):
    # Ensure that the query is a string and not empty
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not input_linked_docs:
        raise ValueError("Linked documents are empty. Please check input.")

    # Initialize Chroma client and collection
    client = chromadb.Client()# Initialize Chroma client
    collections = client.list_collections()  # List all collections
    if "documents" not in collections:
        collection = client.create_collection("documents")
    else:
        collection = client.get_collection("documents")  # Or reuse the existing collection

    #collection = client.create_collection("documents")  # Create a collection in Chroma

    # Use OpenAIEmbeddings for embedding documents
    embedder = OpenAIEmbeddings()

    # Ensure the documents are in the correct format (Document objects)
    documents = [Document(page_content=doc.page_content, metadata={"source": file_path}) for doc in input_linked_docs if
                 isinstance(doc.page_content, str) and doc.page_content.strip()]

    if not documents:
        raise ValueError("No valid documents found for embedding.")

    # Embed the documents
    embeddings = embedder.embed_documents([doc.page_content for doc in documents])

    if not embeddings or len(embeddings) != len(documents):
        raise ValueError("Embeddings are empty or mismatched. Check the input documents and embedder.")

    ids = [str(i) for i in range(len(documents))]
    collection.add(ids=ids, documents=[doc.page_content for doc in documents], embeddings=embeddings)

    # Now use Chroma as the retriever
    retriever = Chroma.from_documents(documents, embedder).as_retriever()

    # Use the retriever with the QA chain
    qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

    # Use invoke instead of run
    result = qa_chain.invoke(query)

    # Extract the answer and sources from the result
    qa_answer = result.get('answer', 'No answer found')
    sources = result.get('sources', 'No sources found')

    return qa_answer, sources

if process_file_clicked:
    st.session_state.graph = nx.Graph()  # Reset graph
    st.session_state.db = None
    st.session_state.qa_chain = None
    main_placeholder.text("Processing uploaded file...✅✅✅")

    # Load PDF data
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    logger.debug(f"Data loaded: {pages}")

    # Clean and concatenate pages
    main_placeholder.text("Cleaning extracted text...✅✅✅")
    cleaned_pages = [clean_text(page.page_content) for page in pages]
    cleaned_text = " ".join(cleaned_pages)

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=500,
        chunk_overlap=50
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    chunks = text_splitter.split_text(cleaned_text)
    chunked_docs = [Document(page_content=chunk, metadata={"source": file_path}) for chunk in chunks]

    # Store docs in session state
    st.session_state.docs = chunked_docs

    # Build document graph
    build_document_graph(chunked_docs, st.session_state.graph)

# Main Query Handling
user_query = main_placeholder.text_input("Question:", st.session_state.query)  # Renamed to user_query

if user_query:  # Use user_query instead of query
    st.session_state.query = user_query  # Update session state with the query

    if st.session_state.docs is None:
        st.error("No documents are loaded. Please upload a file first.")
    else:
        # Refine the query
        if user_query.strip():  # Check for valid query
            refined_query = rewrite_query(user_query)  # Pass user_query to the rewrite function
            st.session_state.refined_query = refined_query  # Save refined query to session state
            st.write(f"Refined Query in Session State: {st.session_state.refined_query}")  # Debugging
        else:
            st.warning("Please enter a valid query.")

        # Check if the session state contains refined_query and docs
        if not st.session_state.refined_query:
            st.warning("Refined query is empty!")
        if not st.session_state.docs:
            st.warning("Documents are empty!")

        # Retrieve relevant documents
        linked_docs = retrieve_linked_documents(st.session_state.graph, st.session_state.refined_query, st.session_state.docs)
        st.session_state.linked_docs = linked_docs  # Save linked docs to session state
        st.write(f"Linked Docs in Session State: {st.session_state.linked_docs}")  # Debugging

        if linked_docs:
            # Generate the answer
            st.write(f"Refined Query: {st.session_state.refined_query.content}")  # Debugging statement

            answer = generate_answer(st.session_state.refined_query.content, linked_docs)
            st.header("Answer")
            st.write(answer)
        else:
            st.error("No relevant documents found.")
