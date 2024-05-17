# Import Libraries
import pandas as pd
import numpy as np
import sqlite3


def create_table():
    
    # Connect to SQLite database (creates a new database if it doesn't exist)
    conn = sqlite3.connect('chroma_vectors.db')

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Create a table to store text chunks and their embeddings
    cursor.execute('''CREATE TABLE IF NOT EXISTS text_chunks (
                        id INTEGER PRIMARY KEY,
                        pdf_name TEXT,
                        chunk_index INTEGER,
                        text TEXT,
                        embedding BLOB
                    )''')

    # Commit changes and close connection
    conn.commit()
    conn.close()