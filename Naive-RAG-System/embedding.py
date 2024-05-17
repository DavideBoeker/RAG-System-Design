# Import Libraries
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# # Ensure the model is in evaluation mode
# model.eval()

def get_bert_embeddings(chunks):

    pass

    # embeddings = []

    # for chunk in chunks:
    #     # Tokenize input text and convert to tensor format
    #     inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
    #     with torch.no_grad():  # Disable gradient calculation for inference
    #         outputs = model(**inputs)
        
    #     # The last hidden state is at index 0, containing embeddings for each token
    #     # To get sentence embeddings, we can use the output of the [CLS] token
    #     cls_embeddings = outputs.last_hidden_state[:, 0, :]

    #     # Append cls embeddings to embeddings list
    #     embeddings.append(cls_embeddings)
    
    # return embeddings
