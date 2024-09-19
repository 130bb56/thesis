import time
import torch
from transformers import BertTokenizer, BertModel
from main import replace_bert_self_attention
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*")

# Function to benchmark model inference and compare outputs
def benchmark_model(model, tokenizer, text, num_iterations=10):
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}  # Use model's device

    # Warm-up
    with torch.no_grad():
        model(**inputs)

    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            outputs = model(**inputs)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time, outputs.last_hidden_state  # Return the last hidden state for comparison

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Original model
original_model = BertModel.from_pretrained('bert-base-uncased').to(device)
original_model.eval()  # Set to evaluation mode
original_time, _ = benchmark_model(original_model, tokenizer, "This is a sample input.")
print(f"Original Model Average Inference Time: {original_time:.6f} seconds")

# Modified model
modified_model = BertModel.from_pretrained('bert-base-uncased').to(device)
replace_bert_self_attention(modified_model)
modified_model.eval()
modified_time, _ = benchmark_model(modified_model, tokenizer, "This is a sample input.")
print(f"Modified Model Average Inference Time: {modified_time:.6f} seconds")
