# Import Libraries
import pandas as pd
import numpy as np
import sqlite3



# Import Modules
import modules.load_text_from_pdf as load_text_from_pdf
import modules.preprocess as preprocess
import modules.embedding as embedding
import modules.chroma_db_operations as chroma_db_operations


# Step 1: Load Data from PDF

### Path to the PDF file
pdf_file_path = '/input_data/test.pdf'

### Extract text from the PDF
# pdf_text = load_text_from_pdf.extract_text_from_pdf(pdf_file_path)


# Step 2: Preprocess the Text Data
# preprocessed_text = preprocess.preprocess_text(pdf_text)

### Split the text into text chunks
# chunks = preprocess.split_text_into_chunks(pdf_text, chunk_length=500, overlap=100)


# # Step 3: Contextual Embedding of Text Data
# embeddings = embedding.get_bert_embeddings(chunks)


# Step 4: Load Text Data to Vector DB

### Define database path and collection name
database_path = ''  # Relative path to the database file
collection_name = 'baba_1'

# collection = vector_db_operations.create_collection(collection_name=collection_name)

# vector_db_operations.add_to_collection(documents=chunks, collection=collection)

query = "Tell me about the revenue growth and what contributed to it."

chroma_db_operations.query_collection(query=query, collection_name=collection_name)
