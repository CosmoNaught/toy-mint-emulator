import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from orderly.run import orderly_run
import orderly
from models import FFNN, GRU, LSTM, BiRNN
from config import NeuralNetConfig
from data_handler import MalariaDataset
from train_eval import train_and_evaluate
from predict import predict
from metrics import calculate_metrics
from save_results import save_results

def main():
    # Configuration and paths
    config = NeuralNetConfig()

    dataframe = pd.read_pickle("data.pkl")
    print("Data Loaded")

    results = []

    # Define your experimental setup
    devices = ["cpu"]#, "cuda"]
    worker_options = [1]#, 2, 4]#, 8]#[2**i for i in range(0, 4)]
    epochs_options = [16]#, 64, 128]#[2**i for i in range(3, 8)]
    batch_sizes = [4096]#, 512, 4096]#[2**i for i in range(5, 15)]
    targeted_training_sizes = [65536]#, 131072] #,262144,524288]#[2**i for i in range(9, 19)]
    repetitions = 1

    # Mapping of neural network types to their classes
    net_classes = {
        "FFNN": FFNN
        # "GRU": GRU,
        # "LSTM": LSTM,
        # "BiRNN": BiRNN
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
                                print(config)

                                # Adjust dataset loading based on the targeted training size
                                dataset = MalariaDataset(dataframe, config.input_size)
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
                                save_results(results)

    # Optionally, save final results with a different name to indicate completion
    final_results_df = pd.DataFrame(results)
    final_results_df.to_csv('experiment_results_final.csv', index=False)
    print("Final experiment results saved.")