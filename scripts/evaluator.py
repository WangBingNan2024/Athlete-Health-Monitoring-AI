import torch
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(test_data, model):
    """Evaluate the trained model on the test data."""
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    predictions, actuals = [], []

    with torch.no_grad():
        for eeg_data, hr_data, movement_data, labels in test_loader:
            # Forward pass
            outputs = model(eeg_data, hr_data, movement_data)

            predictions.append(outputs)
            actuals.append(labels)

    # Concatenate results
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()

    # Calculate evaluation metrics
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")


# Example usage
if __name__ == "__main__":
    # Load test data
    test_data = torch.load("test_data.pth")  # Replace with your actual test data path

    model = torch.load("trained_model.pth")  # Replace with your actual trained model path
    evaluate_model(test_data, model)
