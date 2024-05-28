# Import Libraries
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import re


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
    # print(prompt)

    return prompt


def model_inference(prompt, relevant_chunks):

    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in relevant_chunks]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    return formatted_response


def print_answer(formatted_answer):

    # Define a regular expression pattern to match the content within single quotes
    pattern = r"'(.*?)'"

    # Search for the pattern in the answer string
    match = re.search(pattern, formatted_answer)

    print()
    print()

    if match:
        content = match.group(1)  # Extract the content from the first capturing group
        print("Answer: ", content)
    else:
        print("Content not found in the answer.")

    print()
    print()