# Import Libraries
import pandas as pd
import chromadb

# Import Python Modules
from modules import embedding




def create_collection(collection_name):

    """
    Create a new collection in Chroma DB.

    Parameters:
    collection_name (string): The name of the collection to be created.

    Returns:
    None.
    """

    # Import embedding function
    embedding_function = embedding.BERTEmbeddingFunction()

    # Create chroma client
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)

    # Create the collection
    chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)



def add_to_collection(documents, collection_name):

    """
    Add a list of documents (text chunks) to an existing collection.

    Parameters:
    documents (list): A list of documents (text chunks) that are to be added to the Chroma DB.
    collection_name (string): The name of the collection in which the documents should be added.

    Returns:
    None.
    """

    # Start Chroma client
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)

    # Retrieve the embedding function from the Python module
    embedding_function = embedding.BERTEmbeddingFunction()

    # Fetch the required collection from the Chroma DB
    collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)

    # Create IDs for each document in the document collection
    document_ids = list(map(lambda tup: f"id{tup[0]}", enumerate(documents))) # Simple numeric ID creation

    # Add the documents and IDs to the collection
    collection.add(documents=documents, ids=document_ids)


def query_collection(query, collection_name, required_results):

    """
    Query the Chroma DB to retrieve the most similar text chunks based on the input query.

    Parameters:
    query (text): The query text for which the most similar text chunks should be exported.
    collection_name (string): The name of the collection from which the text chunks should be exported.
    required_results (int): The number of results that should be retrieved from the Chroma DB.

    Returns:
    df (DataFrame): The dataframe that contains the IDs, distances and documents (text chunks) that are most similar to the input query.
    """

    # Start Chroma client
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)

    # Retrieve the embedding function from the Python module
    embedding_function = embedding.BERTEmbeddingFunction()

    # Fetch the required collection from the Chroma DB
    collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)

    # Query the collection and retrieve the top N most similar results compared to the input query
    result = collection.query(query_texts=[query], n_results=required_results, include=["documents", "distances"])

    # Create a DataFrame from the retrieved results and flatten the lists
    df = pd.DataFrame({
        'id': result['ids'][0],
        'distance': result['distances'][0],
        'document': result['documents'][0]
    })

    return df


def delete_collection(collection_name):

    """
    Delete collections from the Chroma DB that are not required anymore.

    Parameters:
    collection_name (string): The name of the collection that should be deleted.

    Returns:
    None.
    """

    # Start the Chroma DB client
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)

    # Delete the collection from the Chroma DB
    chroma_client.delete_collection(name=collection_name)


def create_context(relevant_chunks):
    # Concatenate the text from the document column
    context = " ".join(relevant_chunks["document"].tolist())
    return context