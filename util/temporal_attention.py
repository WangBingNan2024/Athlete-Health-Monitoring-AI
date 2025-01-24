import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(TemporalAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size, attention_size))
        self.attention_bias = nn.Parameter(torch.Tensor(attention_size))
        self.attention_output = nn.Parameter(torch.Tensor(attention_size, 1))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.attention_weights)
        nn.init.zeros_(self.attention_bias)
        nn.init.xavier_uniform_(self.attention_output)

    def forward(self, input_sequence):
        # Compute attention scores
        attention_scores = torch.matmul(input_sequence, self.attention_weights) + self.attention_bias
        attention_scores = F.tanh(attention_scores)  # Apply activation function

        # Compute the attention weights
        attention_weights = torch.matmul(attention_scores, self.attention_output)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum of the input sequence
        attended_output = torch.sum(attention_weights * input_sequence, dim=1)
        return attended_output


# Example usage
if __name__ == "__main__":
    model = TemporalAttention(hidden_size=256, attention_size=128)
    input_sequence = torch.randn(10, 50, 256)  # Example input sequence (batch_size, sequence_length, hidden_size)

    output = model(input_sequence)
    print(output)
