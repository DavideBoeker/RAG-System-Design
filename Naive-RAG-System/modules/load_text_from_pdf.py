# Import Libraries
import PyPDF2


def extract_text_from_pdf(pdf_file_path):

    """
    Function to extract the text from each page in a PDF document and append it to a single text block.

    Parameters:
    pdf_file_path (string): input file path for the PDF document relative to the location of the executing function.

    Returns:
    text (text): Text block element that contains the complete text from the PDF document.
    """

    text = ''
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text