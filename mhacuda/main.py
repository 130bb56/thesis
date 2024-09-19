import torch
from transformers import BertModel, BertConfig
from custom_attention import CustomSelfAttention

def replace_bert_self_attention(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.linear.Linear):
            continue  # Skip Linear layers
        if 'attention.self' in name:
            parent = get_parent_module(model, name)
            setattr(parent, 'self', CustomSelfAttention(model.config))

def get_parent_module(model, module_name):
    modules = module_name.split('.')
    parent = model
    for module in modules[:-1]:
        parent = getattr(parent, module)
    return parent

# Load pretrained BERT model
model = BertModel.from_pretrained('bert-base-uncased')

# Replace attention layers
replace_bert_self_attention(model)

