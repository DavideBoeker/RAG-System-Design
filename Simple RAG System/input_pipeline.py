# Import Libraries
import pandas as pd
import numpy as np
import sqlite3



# Import Modules
import load_text_from_pdf
import preprocess
import embedding


# Step 1: Load Data from PDF

# Path to the PDF file
pdf_file_path = 'test.pdf'

# Extract text from the PDF
pdf_text = load_text_from_pdf.extract_text_from_pdf(pdf_file_path)




# Step 2: Preprocess the Text Data
preprocessed_text = preprocess.preprocess_text(pdf_text)

# Print the extracted text
print(preprocessed_text)

# Step 3: Contextual Embedding of Text Data
embeddings = embedding.get_bert_embeddings(text)


# Step 4: Load Text Data to Vector DB