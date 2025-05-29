import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1) # *2 for bidirectional

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        e = torch.tanh(self.attn(lstm_out)) # (batch_size, seq_len, 1)
        alpha = F.softmax(e, dim=1) # (batch_size, seq_len, 1)
        context = (lstm_out * alpha).sum(dim=1) # (batch_size, hidden_dim * 2)
        return context

class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dense_dim, num_classes, dropout_rate):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2, # As per your notebook
            dropout=dropout_rate if 2 > 1 else 0 # Dropout only between LSTM layers if num_layers > 1
        )
        self.attn = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(hidden_dim * 2, dense_dim) # hidden_dim * 2 because BiLSTM
        self.output_layer = nn.Linear(dense_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.bilstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        context = self.attn(lstm_out)
        # context shape: (batch_size, hidden_dim * 2)
        x = self.dropout(context)
        x = F.relu(self.dense(x))
        logits = self.output_layer(x)
        return logits

if __name__ == '__main__':
    # Example usage
    # These params should match those used in training/data_processing
    test_input_dim = 48 
    test_hidden_dim = 256
    test_dense_dim = 112
    test_num_classes = 8 
    test_dropout_rate = 0.4
    
    model = BiLSTMAttention(test_input_dim, test_hidden_dim, test_dense_dim, test_num_classes, test_dropout_rate)
    
    # Dummy input
    batch_size = 32
    seq_length = 10 # Max sequence length
    dummy_input = torch.randn(batch_size, seq_length, test_input_dim)
    
    output = model(dummy_input)
    print("Model initialized successfully.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be (batch_size, num_classes)