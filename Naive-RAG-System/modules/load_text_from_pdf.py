# Import Libraries
import PyPDF2
import os


def extract_text_from_pdf(pdf_file_path):

    """
    Function to extract the text from each page in a PDF document and append it to a single text block.

    Parameters:
    pdf_file_path (string): input file path for the PDF document relative to the location of the executing function.

    Returns:
    text (text): Text block element that contains the complete text from the PDF document.
    """

    # Retrieve the completed path to the PDF based on the relative path provided through the function input parameter
    current_path = os.getcwd()
    total_file_path = os.path.join(current_path, pdf_file_path)

    text = '' # Initialize an empty text object
    with open(total_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file) # Read the PDF document
        for page in reader.pages:
            text += page.extract_text() # Append the text from each page onto the text block
    return text