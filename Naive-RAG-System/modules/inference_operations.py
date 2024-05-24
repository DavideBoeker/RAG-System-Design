# Import Libraries
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os


# Import Environment variables
access_token = os.getenv('ACCESS_TOKEN') # Access token for Hugging Face Hub



def model_inference(question, context):

    print()
    print()
    print(question)
    print()
    print()
    print(context)
    print()
    print()

    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # Not suitable for VRAM < 16GB
    # model_id = "google/gemma-2b-it"
    # model_id = "microsoft/Phi-3-mini-4k-instruct"
    # model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
    model_id = "google/gemma-1.1-2b-it"

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=access_token, torch_dtype=torch.bfloat16)

    print(model)
    print()
    print()

    # Ensure the eos_token_id is correctly retrieved
    eos_token_id = tokenizer.eos_token_id

    print(eos_token_id)
    print()
    print()

    if eos_token_id is None:
        raise ValueError("The eos_token_id could not be retrieved from the tokenizer.")

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id  # Ensure pad_token_id is set to eos_token_id
    )

    print(generator)
    print()
    print()

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

    prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("")
    ]

    outputs = generator(
        prompt,
        max_new_tokens=256,
        eos_token_id=eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    return outputs[0]["generated_text"][len(prompt):]

