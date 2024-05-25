# Import Libraries
import pandas as pd


# Import Modules
import modules.load_text_from_pdf as load_text_from_pdf
import modules.preprocess as preprocess
import modules.chroma_db_operations as chroma_db_operations


def main():

    # Step 1: Load Data from PDF

    ### Path to the PDF file
    pdf_file_path = 'input_data/test.pdf'

    ### Extract text from the PDF
    pdf_text = load_text_from_pdf.extract_text_from_pdf(pdf_file_path)


    # Step 2: Preprocess the Text Data
    chunks = preprocess.split_text_into_chunks(pdf_text, chunk_length=500, overlap=100)


    # Step 3: Load Text Data to Vector DB

    ### Define database path and collection name
    collection_name = 'test'

    ### Create the Chroma DB collection
    try:
        chroma_db_operations.delete_collection(collection_name=collection_name) # Delete the collection in case it is already existing
    except:
        print("The collection does not exist yet.") # Placeholder statement as this is required by the syntax
    else:
        print("The previous version of the collection has been deleted.")
    finally:
        chroma_db_operations.create_collection(collection_name=collection_name) # The collection is hosted in a Docker container with a local mount in /chroma/

    ### Add the documents (text chunks) to the collection
    chroma_db_operations.add_to_collection(documents=chunks, collection_name=collection_name)



if __name__=="__main__":
    main()
