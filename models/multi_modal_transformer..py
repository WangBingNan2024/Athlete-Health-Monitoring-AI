import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalTransformer(nn.Module):
    def __init__(self, input_size_eeg, input_size_hr, input_size_movement, hidden_size=256, num_heads=8, num_layers=4,
                 output_size=1):
        super(MultiModalTransformer, self).__init__()

        # EEG input processing
        self.eeg_fc = nn.Linear(input_size_eeg, hidden_size)

        # Heart rate input processing
        self.hr_fc = nn.Linear(input_size_hr, hidden_size)

        # Movement data input processing
        self.movement_fc = nn.Linear(input_size_movement, hidden_size)

        # Transformer architecture
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=num_layers
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, eeg_data, hr_data, movement_data):
        # Process each modality
        eeg_out = F.relu(self.eeg_fc(eeg_data))
        hr_out = F.relu(self.hr_fc(hr_data))
        movement_out = F.relu(self.movement_fc(movement_data))

        # Combine modalities (concatenate them)
        combined_input = torch.cat((eeg_out, hr_out, movement_out), dim=-1)

        # Pass through transformer
        transformer_output = self.transformer_encoder(combined_input)

        # Take the last output from transformer and pass through the output layer
        output = self.fc_out(transformer_output[-1])
        return output


# Example usage
if __name__ == "__main__":
    model = MultiModalTransformer(input_size_eeg=64, input_size_hr=1, input_size_movement=6)
    eeg_data = torch.randn(10, 64)  # Example EEG data (batch_size, features)
    hr_data = torch.randn(10, 1)  # Example heart rate data (batch_size, features)
    movement_data = torch.randn(10, 6)  # Example movement data (batch_size, features)

    output = model(eeg_data, hr_data, movement_data)
    print(output)
