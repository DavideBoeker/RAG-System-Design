# Import Libraries
import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def create_prompt(query_text, relevant_chunks):

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in relevant_chunks])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    return prompt


def model_inference(prompt, relevant_chunks):

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in relevant_chunks]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    return formatted_response