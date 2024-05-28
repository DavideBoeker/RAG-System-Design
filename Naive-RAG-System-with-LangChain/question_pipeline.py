# Import Libraries


# Import Modules
from modules import user_interface, chroma_db_operations, inference_operations


# Define Global Variables
CHROMA_PATH = "chroma"



def main():

    # Step 1: Prompt user for question
    question = user_interface.create_cli()


    # Step 2: Retrieve relevant text chunks from Chroma DB
    relevant_chunks = chroma_db_operations.retrieve_relevant_chunks(CHROMA_PATH=CHROMA_PATH, query_text=question)


    # Step 3: Conduct model inference to retrive answer
    prompt = inference_operations.create_prompt(query_text=question, relevant_chunks=relevant_chunks)
    answer = inference_operations.model_inference(prompt=prompt, relevant_chunks=relevant_chunks)

    print()
    print()
    print(answer)
    print()
    print()




if __name__ == "__main__":
    main()