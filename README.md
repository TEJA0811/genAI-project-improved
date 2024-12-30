# Graph-Based Document Processing and Query System

This project enables interactive document processing, building relationships between content using a graph-based approach, and answering user queries through refined embeddings and linked document retrieval. It combines Streamlit for the frontend, FastAPI for the backend, and OpenAI for embeddings and query responses.

## Features

### File Upload and Document Processing:
- Users upload files through the Streamlit interface.
- Documents are cleaned, chunked into manageable pieces, and embedded for further processing.

### Graph-Based Relationship Modeling:
- A graph is built to model relationships between document chunks based on their semantic similarity.
- Helps in retrieving highly relevant sections of content for user queries.

### Query Rewriting and Retrieval:
- Queries are rewritten for specificity using an LLM.
- Relevant document chunks are retrieved based on similarity with the refined query.

### Answer Generation:
- The system generates responses by combining context from retrieved chunks.
- Uses a QA chain powered by OpenAI to ensure contextually accurate answers.

## Architecture Diagram

```mermaid
sequenceDiagram
    participant User
    participant Streamlit_UI
    participant FastAPI_Backend
    participant GraphDB
    participant OpenAI

    User->>Streamlit_UI: Upload file
    Streamlit_UI->>FastAPI_Backend: API call with file
    FastAPI_Backend->>FastAPI_Backend: Process, clean, chunk text
    FastAPI_Backend->>GraphDB: Build graph of document relationships

    User->>Streamlit_UI: Submit query
    Streamlit_UI->>FastAPI_Backend: API call with query
    FastAPI_Backend->>OpenAI: Rewrite query
    FastAPI_Backend->>GraphDB: Retrieve linked chunks
    GraphDB->>FastAPI_Backend: Return relevant chunks
    FastAPI_Backend->>OpenAI: Generate answer with context
    OpenAI->>FastAPI_Backend: Return final answer
    FastAPI_Backend->>Streamlit_UI: Display answer
