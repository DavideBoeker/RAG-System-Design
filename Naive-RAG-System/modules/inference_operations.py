# Import Libraries
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os


# Import Environment variables
access_token = os.getenv('ACCESS_TOKEN') # Access token for Hugging Face Hub


def print_param_dtype(model):
    for name, param in model.named_parameters():
        print(f"{name} is loaded in {param.dtype}")



# def model_inference(question, context):

#     print()
#     print()
#     print(question)
#     print()
#     print()
#     print(context)
#     print()
#     print()

#     # model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # Not suitable for VRAM < 16GB
#     model_id = "microsoft/Phi-3-mini-4k-instruct"
#     # model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
    

#     # Load the tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
#     model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=access_token, torch_dtype=torch.bfloat16)

#     print(model)
#     print()
#     print()

#     # Ensure the eos_token_id is correctly retrieved
#     eos_token_id = tokenizer.eos_token_id

#     print(eos_token_id)
#     print()
#     print()

#     if eos_token_id is None:
#         raise ValueError("The eos_token_id could not be retrieved from the tokenizer.")

#     generator = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         device_map="auto",
#         pad_token_id=tokenizer.eos_token_id  # Ensure pad_token_id is set to eos_token_id
#     )

#     print(generator)
#     print()
#     print()

#     messages = [
#         {   "role": "system", 
#             "content": """You are an advisor. 
#         Answer each question only based on the context given to you. 
#         Do not invent any information beyond the context and say if you don't know the answer.""",
#         },
#         {   "role": "user",
#             "content": f"""Context:
#         {context} 

#         ---

#         Now here is the question you need to answer.

#         Question: {question}""",
#         },
#     ]

#     prompt = tokenizer.apply_chat_template(
#             messages, 
#             tokenize=False, 
#             add_generation_prompt=True
#     )

#     terminators = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("")
#     ]

#     outputs = generator(
#         prompt,
#         max_new_tokens=256,
#         eos_token_id=eos_token_id,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )

#     return outputs[0]["generated_text"][len(prompt):]



# def model_inference(question, context):

#     # model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
#     # model_id = "google/gemma-2b-it"
#     model_id = "google/gemma-1.1-2b-it"
    

#     # Load the tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
#     model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=access_token, device_map="auto", torch_dtype=torch.bfloat16)

#     # Prepare the input text as a single string
#     input_text = (
#         "You are an advisor. Answer each question only based on the context given to you. "
#         "Do not invent any information beyond the context and say if you don't know the answer.\n\n"
#         "Context:\n"
#         f"{context}\n\n"
#         "---\n\n"
#         "Now here is the question you need to answer.\n\n"
#         f"Question: {question}\n\n"
#         "Answer:"
#     )

#     # Tokenize the input text
#     inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

#     # Generate the output
#     outputs = model.generate(
#         inputs['input_ids'],
#         max_new_tokens=256,  # Adjust this as needed
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,  # To avoid padding issues
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )

#     # Decode the generated text
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Post-process the generated text to extract the answer
#     answer_start = generated_text.find("Answer:")
#     if answer_start != -1:
#         generated_text = generated_text[answer_start + len("Answer:"):].strip()

#     return generated_text


def model_inference(question, context):

    model_id = "google/gemma-1.1-2b-it"

    # Load the tokenizer and model with the access token
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=access_token, device_map="auto", torch_dtype=torch.bfloat16)
    # print()
    # print()
    # print_param_dtype(model)
    # print()
    # print()

    # Prepare the input text
    input_text = f"Answer the following question based on the context provided. Question: {question}. Context: {context}."
    # print(f"Input text: {input_text}")

    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    # print(f"Tokenized input IDs: {input_ids}")

    # Generate the output
    print("Generating response...")
    outputs = model.generate(
        input_ids,
        max_new_tokens=60,  # Reduce the number of tokens to generate for faster execution
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # To avoid padding issues
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process the generated text to extract the answer
    answer_start = generated_text.find("Answer:")
    if answer_start != -1:
        generated_text = generated_text[answer_start + len("Answer:"):].strip()
    
    return generated_text
