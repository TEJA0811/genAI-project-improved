from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from langchain.docstore.document import Document

# Placeholder LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")


def rewrite_query(query):
    """
    Rewrites the query using a predefined template to make it more focused.
    """
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Rephrase this query: {query}."
    )
    refined_query = prompt.format(query=query)
    return refined_query


def retrieve_documents(refined_query, graph: nx.Graph, documents: list):
    """
    Retrieves linked documents based on a similarity threshold using the document graph.
    """
    embedder = OpenAIEmbeddings()
    query_embedding = embedder.embed_query(refined_query)

    results = []
    for node in graph.nodes:
        content = documents[node].page_content
        node_embedding = embedder.embed_query(content)
        similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
        if similarity > 0.6:  # Adjusted similarity threshold
            results.append((node, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    linked_docs = [documents[node_id] for node_id, _ in results]

    return linked_docs


def handle_query(query, graph, documents):
    """
    Handles the entire query process: rewrites the query, retrieves documents, and generates an answer.
    """
    refined_query = rewrite_query(query)

    # Retrieve relevant documents
    linked_docs = retrieve_documents(refined_query, graph, documents)

    # If no linked docs are found, raise an error
    if not linked_docs:
        return {"error": "No relevant documents found for the query."}

    # Use Chroma as a retriever
    retriever = Chroma.from_documents(linked_docs, OpenAIEmbeddings()).as_retriever()

    # Create a QA chain
    qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

    # Generate answer
    result = qa_chain.invoke(refined_query)

    return {
        "answer": result.get("answer", "No answer found"),
        "sources": result.get("sources", "No sources found"),
    }
