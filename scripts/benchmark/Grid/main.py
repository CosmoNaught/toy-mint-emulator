import os
import csv
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Model Definition
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
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, dropout=dropout_prob)  # Dropout between RNN layers
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout before the FC layer
        self.softplus = nn.Softplus()

    def forward(self, x):
        x, _ = self.gru(x.view(len(x), 1, -1))
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc(x.view(len(x), -1))
        x = self.softplus(x)
        return x

    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=dropout_prob)  # Dropout between LSTM layers
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout before the FC layer
        self.softplus = nn.Softplus()

    def forward(self, x):
        x, _ = self.lstm(x.view(len(x), 1, -1))
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc(x.view(len(x), -1))
        x = self.softplus(x)
        return x


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True, dropout=dropout_prob)  # Dropout between RNN layers
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.ln = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout before the FC layer
        self.softplus = nn.Softplus()

    def forward(self, x):
        x, _ = self.rnn(x.view(len(x), 1, -1))
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc(x.view(len(x), -1))
        x = self.softplus(x)
        return x



# Neural Network Settings
@dataclass
class NeuralNetConfig:
    epochs: int = 4
    batch_size: int = 4096
    hidden_size: int = 64
    dropout_prob: float = 0.5
    shuffle: bool = True
    num_workers: int = 2
    test_pct: float = 0.2
    val_pct: float = 0.2
    input_size: int = 20
    output_size: int = 61

# Malaria Dataset
class MalariaDataset(Dataset):
    def __init__(self, csv_file, input_size):
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df.iloc[:, :input_size].values, dtype=torch.float32)
        self.Y = torch.tensor(df.iloc[:, input_size:].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train_and_evaluate(model, train_loader, val_loader, test_loader, device, criterion, optimizer, scheduler, config):
    start_time = time.time()  # Start timing the training process

    train_data_loading_times = []
    train_processing_times = []
    val_data_loading_times = []
    val_processing_times = []

    for epoch in tqdm(range(config.epochs), desc='Training Progress'):
        model.train()
        train_loss = 0

        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} - Training Data Loading', leave=False):
            data_load_start = time.time()
            data, target = data.to(device), target.to(device)
            data_load_end = time.time()

            process_start = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            process_end = time.time()

            train_data_loading_times.append(data_load_end - data_load_start)
            train_processing_times.append(process_end - process_start)

        train_loss /= len(train_loader.dataset)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} - Validation Data Loading', leave=False):
                data_load_start = time.time()
                data, target = data.to(device), target.to(device)
                data_load_end = time.time()

                process_start = time.time()
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                process_end = time.time()

                val_data_loading_times.append(data_load_end - data_load_start)
                val_processing_times.append(process_end - process_start)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

    total_training_time = time.time() - start_time

    # Calculating averages
    avg_train_data_load_time = sum(train_data_loading_times) / len(train_data_loading_times)
    avg_train_process_time = sum(train_processing_times) / len(train_processing_times)
    avg_val_data_load_time = sum(val_data_loading_times) / len(val_data_loading_times)
    avg_val_process_time = sum(val_processing_times) / len(val_processing_times)

    test_loss = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing Data Loading', leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    
    return total_training_time, avg_train_data_load_time, avg_train_process_time, avg_val_data_load_time, avg_val_process_time, test_loss


def predict(model, loader, device):
    predictions = []
    actual = []
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
            actual.append(target.cpu().numpy())
    return predictions, actual

def calculate_metrics(predictions, actual):
    # Convert lists of arrays into single numpy arrays
    predictions = np.vstack(predictions)
    actual = np.vstack(actual)
    
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)
    
    return mae, mse, rmse, r2

def main():
    # Configuration and paths
    config = NeuralNetConfig()
    csv_file_path = 'D:\\Malaria\\toy-mint-emulator\\data\\mint_data_scaled.csv'
    results = []

    # Define your experimental setup
    devices = ["cpu", "cuda"]
    worker_options = [1, 2] #, 4, 8]#[2**i for i in range(0, 4)]
    epochs_options = [32, 64] #, 128]#[2**i for i in range(3, 8)]
    batch_sizes = [64, 512] #, 4096]#[2**i for i in range(5, 15)]
    targeted_training_sizes = [65536, 131072] #,262144,524288]#[2**i for i in range(9, 19)]
    repetitions = 2

    # Mapping of neural network types to their classes
    net_classes = {
        "FFNN": FFNN,
        "GRU": GRU,
        "LSTM": LSTM,
        "BiRNN": BiRNN
    }

    for device_name in devices:
        for workers in worker_options:
            for epochs in epochs_options:
                for batch_size in batch_sizes:
                    for training_size in targeted_training_sizes:
                        for net_type in net_classes.keys():  # Added loop
                            for repetition in range(repetitions):
                                device = torch.device(device_name if torch.cuda.is_available() else "cpu")
                                config.num_workers = workers
                                config.epochs = epochs
                                config.batch_size = batch_size

                                # Adjust dataset loading based on the targeted training size
                                dataset = MalariaDataset(csv_file_path, config.input_size)
                                val_size = int(training_size * config.val_pct / (1 - config.test_pct - config.val_pct))
                                test_size = int(training_size * config.test_pct / (1 - config.test_pct - config.val_pct))
                                total_size_needed = training_size + val_size + test_size
                                # Optionally ensure your dataset is large enough or handle smaller datasets
                                dataset, _ = random_split(dataset, [total_size_needed, len(dataset) - total_size_needed])

                                train_dataset, val_dataset, test_dataset = random_split(dataset, [training_size, val_size, test_size])
                                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=True)
                                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
                                test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

                                # Instantiate the model based on the net type
                                model_class = net_classes[net_type]
                                model = model_class(config.input_size, config.hidden_size, config.output_size, config.dropout_prob).to(device)
                                optimizer = optim.Adam(model.parameters())
                                criterion = nn.MSELoss()
                                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
                                
                                total_training_time, avg_train_data_load_time, avg_train_process_time, avg_val_data_load_time, avg_val_process_time, test_loss = train_and_evaluate(model, train_loader, val_loader, test_loader, device, criterion, optimizer, scheduler, config)
                                predictions, actual = predict(model, test_loader, device)
                                mae, mse, rmse, r2 = calculate_metrics(predictions, actual)
                                
                                results.append({
                                    "neural_net_type": net_type,  # Added attribute
                                    "repetition": repetition + 1,
                                    "device": device_name,
                                    "num_workers": workers,
                                    "nn_epochs": epochs,
                                    "nn_batch_size": batch_size,
                                    "targeted_training_size": training_size,
                                    "total_training_time": total_training_time,
                                    "avg_train_data_load_time": avg_train_data_load_time,
                                    "avg_train_process_time": avg_train_process_time,
                                    "avg_val_data_load_time": avg_val_data_load_time,
                                    "avg_val_process_time": avg_val_process_time,
                                    "test_loss": test_loss,
                                    "mae": mae,
                                    "mse": mse,
                                    "rmse": rmse,
                                    "r2": r2
                                })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('tests\\experiment_results.csv', index=False)
    print("Experiment results saved.")

if __name__ == "__main__":
    main()