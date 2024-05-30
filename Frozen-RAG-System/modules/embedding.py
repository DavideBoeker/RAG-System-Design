# Import Libraries
import torch
from transformers import BertTokenizer, BertModel


class BERTEmbeddingFunction:

    """
    A class used to create a custom embedding function for contextual text embedding.
    The structure of the function in this class is according to the requirements for the Chroma DB operation functions.
    """

    def __init__(self, model_name='bert-base-uncased'):

        """
        Initialize the tokenizer and the model used for the contextual text embedding.

        Parameters:
        model_name (text): The name of the model used for the text embedding from the Hugging Face model zoo.
        """

        self.tokenizer = BertTokenizer.from_pretrained(model_name) # Initialize the tokenizer used in the embedding function
        self.model = BertModel.from_pretrained(model_name) # Initialize the model used in the embedding function
        
    def __call__(self, input):

        """
        Execute the contextual text embedding using the previously initialized tokenizer and model.

        Parameters:
        input (list): A list of documents (text chunks) for which the embedding vector should be created.

        Returns:
        embeddings (list): The list containing the embedding vectors for each element from the input list.
        """

        inputs = self.tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=512) # Tokenize the each element from the input list of documents

        with torch.no_grad():
            outputs = self.model(**inputs) # Run the embedding model on the tokenized input data
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Take the mean of the hidden states to obtain a fixed size, single vector representation (embedding)

        return embeddings.tolist()  # Convert tensors to list to match the later required input format for the vector DB