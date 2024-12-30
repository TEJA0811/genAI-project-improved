import re
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

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

def retrieve_linked_documents(query):
    # Implement your document graph logic or use dummy logic
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
    #return []  # Placeholder
