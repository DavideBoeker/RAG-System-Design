# Import Libraries


# Import Modules
from modules import load_text_from_pdf, preprocess_text, chroma_db_operations


# Define Global Variables
DATA_PATH = "input_data"
CHROMA_PATH = "chroma"


def main():

    # Step 1: Load text from PDF file
    documents = load_text_from_pdf.load_documents(DATA_PATH=DATA_PATH)

    # Step 2: Preprocess the text
    chunks = preprocess_text.split_text(documents=documents)

    # Step 3: Load text to Chroma DB
    chroma_db_operations.save_to_chroma(CHROMA_PATH=CHROMA_PATH, chunks=chunks)



if __name__ == "__main__":
    main()