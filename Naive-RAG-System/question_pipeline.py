# Import Libraries



# Import Python Modules
from modules import chroma_db_operations
from modules import inference_operations


def main():

    # Step 1: Receive question
    question = "What is the revenue growth this year and what contributed to it?"


    # Step 2: Retrieve relevant Text Chunks from Chroma DB
    collection_name = "test"
    relevant_chunks = chroma_db_operations.query_collection(query=question, collection_name=collection_name, required_results=3)


    # Step 3: Retrieve Answer to Question from LLM Inference
    answer = inference_operations.model_inference(question=question, context=relevant_chunks)
    print(answer)



if __name__=="__main__":
    main()