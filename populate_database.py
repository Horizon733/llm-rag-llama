import uuid
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from bs4 import BeautifulSoup as Soup
from qdrant_client import QdrantClient
from qdrant_client.grpc import PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException

from get_embedding_function import get_embedding_function


def extract_website(url: str, max_depth: int = 2, embeddings: Embeddings = None) -> List[Document]:
    """Extract webpages from a given url."""
    print(f"Extracting URLs from: {url}")

    loader = RecursiveUrlLoader(
        url,
        max_depth=max_depth,
        extractor=lambda x: Soup(x, "html.parser").text,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        check_response_status=True
    )

    documents = loader.load()
    print(f"Visited URLs: {url}")
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def populate_chroma_db(
        embeddings: Embeddings,
        docs: List[Document],):
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="chroma"
    )
    vectordb.persist()


def main():
    embeddings = get_embedding_function()
    docs = extract_website(
        url="https://horizon733.github.io/",
        max_depth=2,
        embeddings=embeddings
    )
    populate_chroma_db(
        docs=docs,
        embeddings=embeddings
    )


if __name__ == "__main__":
    main()

