import torch

from llama_index.llms import LlamaCPP
# from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt

# Using a smaller model URL (example: a smaller version of the Mistral model or any other smaller model you find suitable)
model_url = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf'

llm = LlamaCPP(
    # Model URL for downloading automatically
    model_url=model_url,
    # Optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # Adjust context window as needed
    context_window=2048,  # Reducing context window to fit within CPU limits
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # Set n_gpu_layers to 0 to ensure CPU usage
    model_kwargs={"n_gpu_layers": 0},
    # Transform inputs into Llama2 format
    # messages_to_prompt=messages_to_prompt,
    # completion_to_prompt=completion_to_prompt,
    # verbose=True,
)