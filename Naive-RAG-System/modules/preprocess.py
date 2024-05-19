# Import Libries
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Spacy model for NER and lemmatization
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    # Remove headers, footers, and other non-text elements
    # This step is highly dependent on your specific PDF format
    # Example: text = re.sub(r'Header.*\n', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back to string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


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