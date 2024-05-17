# Import Libraries
import pandas as pd
import numpy as np
import sqlite3



# Import Modules
import load_text_from_pdf
import preprocess
import embedding
import vector_db_operations


# Step 1: Load Data from PDF

### Path to the PDF file
pdf_file_path = 'test.pdf'

### Extract text from the PDF
pdf_text = load_text_from_pdf.extract_text_from_pdf(pdf_file_path)


# Step 2: Preprocess the Text Data
preprocessed_text = preprocess.preprocess_text(pdf_text)

### Split the text into text chunks
chunks = preprocess.split_text_into_chunks(pdf_text, chunk_length=500, overlap=100)


# Step 3: Contextual Embedding of Text Data
embeddings = embedding.get_bert_embeddings(chunks)


# Step 4: Load Text Data to Vector DB

### Define database path and collection name
database_path = './database.db'  # Relative path to the database file
collection_name = 'baba_2024'

### Create table in vector DB if it does not exist yet
vector_db_operations.initialize_table(database_path=database_path, collection_name=collection_name)

### Load text chunks, embeddings and metadata to vector DB
vector_db_operations.load_data_to_db(
    text_chunks=chunks,
    embeddings=embeddings,
    pdf_name=pdf_file_path,
    collection_name=collection_name,
    database_path=database_path
)