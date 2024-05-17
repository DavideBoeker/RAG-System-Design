# Import Libraries
import pandas as pd
import numpy as np
import sqlite3



# Import Modules
import load_text_from_pdf
import preprocess
import embedding


# Step 1: Load Data from PDF

### Path to the PDF file
pdf_file_path = 'test.pdf'

### Extract text from the PDF
pdf_text = load_text_from_pdf.extract_text_from_pdf(pdf_file_path)


# Step 2: Preprocess the Text Data
preprocessed_text = preprocess.preprocess_text(pdf_text)

### Split the text into text chunks
chunks = preprocess.split_text_into_chunks(pdf_text, chunk_length=500, overlap=100)

# # Assuming `chunks` is the list of text chunks
# spacing = "\n\n"  # You can adjust the spacing as needed

# # Print the first three chunks with spacing between them
# print(spacing.join(chunks[:3]))


# Step 3: Contextual Embedding of Text Data
embedding.get_bert_embeddings(chunks)

# print(embeddings[:3])


# Step 4: Load Text Data to Vector DB

### Create table in vector DB if it does not exist yet
# create_vector_db.create_table()