# Import Libraries
import torch


# Import Python Modules
from modules import chroma_db_operations
from modules import inference_operations
from modules.execution_diagnostic import execution_time


@execution_time
def main():

    # Step 1: Receive question
    question = "Has the revenue increased in 2024 and what contributed to it?"


    # Step 2: Retrieve relevant Text Chunks from Chroma DB
    collection_name = "test"
    relevant_chunks = chroma_db_operations.query_collection(query=question, collection_name=collection_name, required_results=3)
    context = chroma_db_operations.create_context(relevant_chunks)

    print()
    print()
    print(context)
    print()
    print()


    # Step 3: Retrieve Answer to Question from LLM Inference
    answer = inference_operations.inference_1B_model(question=question, context=context)
    print()
    print()
    print(answer)
    print()
    print()



if __name__=="__main__":
    main()