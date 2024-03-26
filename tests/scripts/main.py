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
from pyDOE2 import lhs

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
    worker_options = [2**i for i in range(0, 4)]  # 4 levels
    epochs_options = [2**i for i in range(3, 8)]  # 5 levels
    batch_sizes = [2**i for i in range(5, 15)]  # 10 levels
    targeted_training_sizes = [2**i for i in range(9, 19)]  # 10 levels
    repetitions = 8

    # Number of samples for LHS - Adjust as needed
    num_samples = 100

    # LHS Sampling
    sample_space = sample_space = lhs(4, samples=num_samples, criterion='cm', iterations=100)

    for device_name in devices:
        for sample in sample_space:
            workers = worker_options[int(np.floor(sample[0] * len(worker_options)))]
            epochs = epochs_options[int(np.floor(sample[1] * len(epochs_options)))]
            batch_size = batch_sizes[int(np.floor(sample[2] * len(batch_sizes)))]
            training_size = targeted_training_sizes[int(np.floor(sample[3] * len(targeted_training_sizes)))]

            for repetition in range(repetitions):
                device = torch.device(device_name if torch.cuda.is_available() else "cpu")
                config.num_workers = workers
                config.epochs = epochs
                config.batch_size = batch_size
                
                # Load and shuffle dataset for each repetition
                dataset = MalariaDataset(csv_file_path, config.input_size)
                total_size = len(dataset)
                
                # Shuffle dataset
                indices = torch.randperm(total_size).tolist()
                dataset = torch.utils.data.Subset(dataset, indices)

                # Calculate new split sizes for this repetition
                val_size = int(training_size * config.val_pct / (1 - config.test_pct - config.val_pct))
                test_size = int(training_size * config.test_pct / (1 - config.test_pct - config.val_pct))
                train_size = training_size
                total_size_needed = train_size + val_size + test_size
                if total_size_needed > total_size:
                    raise ValueError("Dataset is not large enough for the desired splits.")

                # Split dataset
                train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[-test_size:]
                train_dataset = torch.utils.data.Subset(dataset, train_indices)
                val_dataset = torch.utils.data.Subset(dataset, val_indices)
                test_dataset = torch.utils.data.Subset(dataset, test_indices)
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
                test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

                model = FFNN(config.input_size, config.hidden_size, config.output_size, config.dropout_prob).to(device)
                optimizer = optim.Adam(model.parameters())
                criterion = nn.MSELoss()
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
                
                total_training_time, avg_train_data_load_time, avg_train_process_time, avg_val_data_load_time, avg_val_process_time, test_loss = train_and_evaluate(model, train_loader, val_loader, test_loader, device, criterion, optimizer, scheduler, config)
                predictions, actual = predict(model, test_loader, device)
                mae, mse, rmse, r2 = calculate_metrics(predictions, actual)
                
                results.append({
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
    results_df.to_csv('tests\\data\\lhs_experiment_results.csv', index=False)
    print("Experiment results saved.")

if __name__ == "__main__":
    main()