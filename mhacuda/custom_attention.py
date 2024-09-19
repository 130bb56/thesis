import torch
import torch.nn as nn
import mha_cuda  # Import your CUDA extension

class CustomSelfAttention(nn.Module):
    def __init__(self, config):
        super(CustomSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear layers
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        #print(f"hidden_states.device: {hidden_states.device}")
        #print(f"self.query.weight.device: {self.query.weight.device}")
        
        # Ensure all weights are on the same device as hidden_states
        self.query = self.query.to(hidden_states.device)
        self.key = self.key.to(hidden_states.device)
        self.value = self.value.to(hidden_states.device)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Shape transformation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Use custom CUDA MHA
        context_layer = mha_cuda.mha_forward(
            query_layer.contiguous(),
            key_layer.contiguous(),
            value_layer.contiguous()
        )

        # Restore shape
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        if output_attentions:
            attention_probs = torch.zeros(
                hidden_states.size(0),
                self.num_attention_heads,
                hidden_states.size(1),
                hidden_states.size(1),
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            outputs += (attention_probs,)

        return outputs

