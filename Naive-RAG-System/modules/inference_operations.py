# Import Libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os


# Import Environment variables
access_token = os.getenv('ACCESS_TOKEN') # Access token for Hugging Face Hub


def inference_1B_model(question, context):

    model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"

    # Load the tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=access_token)

    print("Creating pipeline...")
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id  # Ensure pad_token_id is set to eos_token_id
    )

    messages = [
        {   "role": "system", 
            "content": """You are an advisor. 
        Answer each question only based on the context given to you. 
        Do not invent any information beyond the context and say if you don't know the answer.""",
        },
        {   "role": "user",
            "content": f"""Context:
        {context} 

        ---

        Now here is the question you need to answer.

        Question: {question}""",
        },
    ]

    print("Creating prompt...")
    prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    print("Generating response...")
    outputs = generator(
        prompt,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    return outputs[0]["generated_text"][len(prompt):]


def inference_2B_model(question, context):

    model_id = "google/gemma-1.1-2b-it"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Create input text for the model
    input_text = f"Answer the following question based on the context provided. Question: {question}. Context: {context}."

    # Tokenize the input text and convert it to tensor
    print("Tokenizing input text...")
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs.input_ids.to(model.device)

    # Generate the output
    print("Generating response...")
    outputs = model.generate(
        input_ids,
        max_new_tokens=60,  # Adjust as needed
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # To avoid padding issues
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text