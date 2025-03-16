import torch
import torch.nn as nn
import torch.optim as optim

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=64*41, hidden_dim=64, num_layers=2, num_classes=40):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state
        batch_size, trajectory_count, seq_length, input_dim = x.shape
        x = x.view(batch_size,  seq_length, trajectory_count* input_dim)  # Merge batch and trajectory

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # print(x.shape)
        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        out = out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        
        # Fully connected layer
        out = self.fc(out)  # Shape: (batch_size, num_classes)
        
        return out



# # Create the model
# model = TrajectoryLSTM(input_dim, hidden_dim, num_layers, num_classes)

# # Dummy input (batch of 20 trajectories, each of length 30 with 10 features)
# x = torch.randn(batch_size, seq_length, input_dim)

# # Forward pass
# outputs = model(x)
# print(outputs.shape)  # Expected: (20, num_classes)
