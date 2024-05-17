# Import Libraries
import PyPDF2


def extract_text_from_pdf(pdf_file_path):
    text = ''
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def split_text_into_chunks(text, chunk_length, overlap):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_length
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move the start position by chunk_length - overlap to achieve the desired overlap
        start += chunk_length - overlap
    
    return chunks