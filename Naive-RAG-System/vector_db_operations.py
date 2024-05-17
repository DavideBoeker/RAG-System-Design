# Import Libraries
from chromadb import HttpClient
from chromadb.utils import embedding_functions
import chromadb

from chromadb import Documents, EmbeddingFunction, Embeddings
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return embeddings


def create_collection(collection_name):

    # ef = embedding_functions.DefaultEmbeddingFunction()
    # client = HttpClient(host="localhost", port=8000)

    # print('HEARTBEAT:', client.heartbeat())

    # collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)

    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collection = chroma_client.create_collection(name=collection_name)

    return collection


def add_to_collection(documents, collection):

    # Every document needs an id for Chroma
    document_ids = list(map(lambda tup: f"id{tup[0]}", enumerate(documents)))

    collection.add(documents=documents, ids=document_ids)


def query_collection(query, collection):

    result = collection.query(query_texts=[query], n_results=5, include=["documents", 'distances',])

    for id_, document, distance in zip(result):
        print(f"ID: {id_}, Document: {document}, Similarity: {1 - distance}")