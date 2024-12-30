import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import networkx as nx

UPLOAD_DIR = Path("uploaded_files")

def clean_text(raw_text):
    cleaned = re.sub(r'\s+', ' ', raw_text)
    cleaned = re.sub(r'Page \d+ of \d+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(\w)-\n(\w)', r'\1\2', cleaned)
    cleaned = re.sub(r'\n', ' ', cleaned)
    return cleaned

def split_text(text, chunk_size=500, overlap=50):
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size - overlap)
    ]

def build_document_graph(docs):
    graph = nx.Graph()
    embeddings = [doc.metadata["embedding"] for doc in docs]
    for i, emb1 in enumerate(embeddings):
        for j, emb2 in enumerate(embeddings):
            if i != j:
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                if similarity > 0.6:
                    graph.add_edge(i, j, weight=similarity)
    return graph

async def process_file(file):
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    cleaned_pages = [clean_text(page.page_content) for page in pages]
    cleaned_text = " ".join(cleaned_pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(cleaned_text)

    return {"chunks": chunks, "graph": build_document_graph(chunks)}
