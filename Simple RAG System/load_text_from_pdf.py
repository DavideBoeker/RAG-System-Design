# Import Libraries
import PyPDF2


def extract_text_from_pdf(pdf_file_path):
    text = ''
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text