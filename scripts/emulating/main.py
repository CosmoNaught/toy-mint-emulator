import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

# Model definition must match the one used for training
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(FFNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.layer3 = nn.Linear(output_size, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.softplus(x)

def main():
	# Load the model
	model_path = 'model/trained_model.pth'  # Adjust the path as necessary
	model = FFNN(input_size=20, hidden_size=64, output_size=61, dropout_prob=0.5)
	model.load_state_dict(torch.load(model_path))
	model.eval()  # Set the model to evaluation mode

	# Assuming the CSV is correctly formatted and matches the model's expected input
	csv_file_path = 'D:\\Malaria\\toy-mint-emulator\\data\\mint_data_scaled.csv'  # Adjust the path as necessary
	df = pd.read_csv(csv_file_path)

	# Select a row (for example, the first row) to emulate
	input_features = df.iloc[0, :20].values  # Adjust as necessary to match input size
	input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

	with torch.no_grad():  # No need to track gradients
		predicted_output = model(input_tensor)

	# Convert the predicted output to a numpy array for plotting
	predicted_output = predicted_output.squeeze().numpy()  # Remove batch dimension and convert to numpy

	# Actual outputs for comparison (if available)
	actual_output = df.iloc[0, 20:].values  # Adjust indices as necessary

	# Plotting
	plt.figure(figsize=(10, 6))
	plt.plot(predicted_output, label='Predicted Output')
	plt.plot(actual_output, label='Actual Output', linestyle='--')
	plt.title('Model Output Comparison')
	plt.xlabel('Output Index')
	plt.ylabel('Output Value')
	plt.legend()
	plt.show()

if __name__ == "__main__":
    main()

