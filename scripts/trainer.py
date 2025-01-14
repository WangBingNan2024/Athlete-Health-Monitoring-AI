import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.multi_modal_transformer import MultiModalTransformer
from models.temporal_attention import TemporalAttention


def train_model(train_data, model, criterion, optimizer, epochs=10, batch_size=64):
    """Train the model using the training data."""
    model.train()
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for eeg_data, hr_data, movement_data, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(eeg_data, hr_data, movement_data)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")


def initialize_model(input_size_eeg, input_size_hr, input_size_movement, hidden_size=256, num_heads=8, num_layers=4,
                     output_size=1):
    """Initialize the model, criterion, and optimizer."""
    model = MultiModalTransformer(input_size_eeg, input_size_hr, input_size_movement, hidden_size, num_heads,
                                  num_layers, output_size)
    criterion = torch.nn.MSELoss()  # For regression tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer


# Example usage
if __name__ == "__main__":
    # Load training data
    train_data = torch.load("train_data.pth")  # Replace with your actual training data path

    model, criterion, optimizer = initialize_model(input_size_eeg=64, input_size_hr=1, input_size_movement=6)
    train_model(train_data, model, criterion, optimizer)
