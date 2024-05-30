# Import Libraries
from langchain.document_loaders import DirectoryLoader

# Import Modules


def load_documents(DATA_PATH):
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents
