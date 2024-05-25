# Import Libraries
import torch
from torch.quantization import quantization
from transformers import AutoModelForCausalLM


def load_quantized_model(quantized_model_path):

    model = torch.load(quantized_model_path)

    return model


def quantize_and_save_model(model_id, quantized_model_path):
    
    print("Loading model for quantization...")
    model = AutoModelForCausalLM.from_pretrained(model_id)

    print("Quantizing the model...")
    quantized_model = quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    print("Saving the quantized model...")
    quantized_model.to('cpu')  # Move to CPU before saving
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print("Quantized model saved.")
    print()