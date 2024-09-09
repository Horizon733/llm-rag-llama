from langchain_community.embeddings import OllamaEmbeddings


def get_embedding_function():
    return OllamaEmbeddings(model="llama3.1")