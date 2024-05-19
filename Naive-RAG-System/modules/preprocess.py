# Import Libries


def split_text_into_chunks(text, chunk_length, overlap):

    """
    Function used to split a text block into text chunks that can be further used by LLMs and derivative systems.

    Parameters:
    text (text): Text block that contains the text which will be split into smaller text chunks.
    chunk_length (int): Parameter to define the length of each text chunk element.
    overlap (int): Parameter to define the overlap between text chunk elements based on the original text block. This overalp is required to ensure context understanding when ingesting the resulting text chunks into LLMs.

    Returns:
    chunks (list): A list containing the previously created text chunks.
    """

    # Initialize a empty list of text chunks as well as counter for iterating through the text block
    chunks = []
    start = 0
    
    # Iterating through the text block one text chunk at a time and appending the chunks to the chunk list
    while start < len(text):
        end = start + chunk_length
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move the start position by chunk_length - overlap to achieve the desired overlap
        start += chunk_length - overlap
    
    return chunks