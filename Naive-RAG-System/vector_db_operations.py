# Import Libraries
from chromadb.api import ChromaDB


def create_database(database_path):
    # Initialize ChromaDB client with SQLite engine
    chromadb = ChromaDB(engine='sqlite', database=database_path)
    
    # Check if the database already exists
    if not chromadb.database_exists():
        # Create the database
        chromadb.create_database()
        print(f"Database created at {database_path}")
    else:
        print(f"Database already exists at {database_path}")


def create_collection(database_path, collection_name, dimensionality):
    # Initialize ChromaDB client with SQLite engine
    chromadb = ChromaDB(engine='sqlite', database=database_path)
    
    # Check if the collection already exists
    if not chromadb.collection_exists(collection_name):
        # Create the collection
        collection = chromadb.create_collection(collection_name, dimension=dimensionality)
        print(f"Collection '{collection_name}' created with dimensionality {dimensionality}")
    else:
        print(f"Collection '{collection_name}' already exists")


def initialize_table(database_path, collection_name):
    
    dimensionality = 768  # Assuming BERT embeddings with dimensionality 768
    
    create_database(database_path)
    create_collection(database_path, collection_name, dimensionality)


def get_chromadb(database_path):
    # Initialize ChromaDB client with SQLite engine
    chromadb = ChromaDB(engine='sqlite', database=database_path)
    return chromadb


def create_metadata(pdf_name, num_chunks):
    metadata = []
    for i in range(num_chunks):
        metadata.append({"pdf_name": pdf_name, "chunk_index": i + 1})
    return metadata


def load_data_to_db(text_chunks, embeddings, pdf_name, collection_name, database_path):

    metadata = create_metadata(pdf_name, num_chunks=len(text_chunks))
    chromadb = get_chromadb(database_path)

    # Delete all data from the collection before loading in the new data
    chromadb[collection_name].delete_all()
   
    # Insert text chunks, embeddings, and metadata into the collection
    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        # Create data dictionary with text chunk, embedding, and metadata
        data = {'text_chunk': chunk, 'embedding': embedding}
        if metadata:
            data['metadata'] = metadata[i] if i < len(metadata) else {}
        
        # Insert data into the collection
        chromadb[collection_name].insert(data)
    
    print(f"{len(text_chunks)} text chunks inserted into collection '{collection_name}'")