# Import Libraries
import pandas as pd
import numpy as np
import sqlite3


# Connect to SQLite database (creates a new database if it doesn't exist)
conn = sqlite3.connect('chroma_vectors.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Create a table to store Chroma vectors
cursor.execute('''CREATE TABLE IF NOT EXISTS chroma_vectors
                  (id INTEGER PRIMARY KEY,
                  vector_name TEXT,
                  chroma_vector TEXT,
                  metadata TEXT)''')

# Commit changes and close connection
conn.commit()
conn.close()