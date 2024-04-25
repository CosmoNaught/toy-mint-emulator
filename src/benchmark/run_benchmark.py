import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools

from pyDOE2 import lhs
from models import FFNN, GRU, LSTM, BiRNN
from config import NeuralNetConfig
from data_handler import MalariaDataset
from train_eval import train_and_evaluate
from predict import predict
from metrics import calculate_metrics
from save_results import save_results
from lhs_method import generate_lhs_samples

def main():

    # Setting seeds for different libraries
    random.seed(42)       # Python's built-in random module
    np.random.seed(42)    # NumPy library
    torch.manual_seed(42) # PyTorch for CPU operations
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(42)  # PyTorch for all CUDA devices (GPUs)
    
    # Configuration and paths
    config = NeuralNetConfig()
    dataframe = pd.read_pickle("data.pkl")
    results = []
    current_test = 0
    devices = ["cpu", "cuda"]
    
    # Parameter ranges
    thread_counts = [2**i for i in range(0, 7)]
    worker_counts = [2**i for i in range(0, 7)]
    epochs_options = [2**i for i in range(3, 8)]
    batch_sizes = [2**i for i in range(5, 15)]
    training_sizes = [2**i for i in range(9, 19)]
    net_classes = {"FFNN": FFNN, "GRU": GRU, "LSTM": LSTM, "BiRNN": BiRNN}
    repetitions = 2

    use_lhs = True  # Toggle this to switch between LHS and full grid search

    if use_lhs:
        # Define dimensions lengths for LHS (excluding device, network type, and repetitions)
        dimensions = [len(thread_counts), len(worker_counts), len(epochs_options), len(batch_sizes), len(training_sizes)]
        num_samples = 100  # Define the number of LHS samples you want

        # Generate LHS samples
        lhs_samples = generate_lhs_samples(dimensions, num_samples)
        parameter_combinations = lhs_samples
    else:
        # Full grid search
        parameter_combinations = list(itertools.product(thread_counts, worker_counts, epochs_options, batch_sizes, training_sizes))
        num_samples = len(parameter_combinations)  # This now represents the total combinations in grid search

    # Calculate total number of tests
    total_tests = num_samples * len(devices) * len(net_classes) * repetitions

    # Timing overall tests
    start_all_tests_time = time.time()

    # Iterate over parameter combinations
    for combination in parameter_combinations:
        if use_lhs:
            num_threads, num_workers, epochs, batch_size, training_size = [
                thread_counts[combination[0]],
                worker_counts[combination[1]],
                epochs_options[combination[2]],
                batch_sizes[combination[3]],
                training_sizes[combination[4]],
            ]
        else:
            num_threads, num_workers, epochs, batch_size, training_size = combination

        # Ensure DataLoader's num_workers does not exceed the set number of threads
        num_workers = min(num_workers, num_threads)

        torch.set_num_threads(num_threads)  # Set the number of threads for PyTorch processes

        for device_name in devices:
            for net_type, model_class in net_classes.items():
                for rep in range(repetitions):
                    start_test_time = time.time()
                    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

                    # Adjust dataset loading based on the targeted training size
                    dataset = MalariaDataset(dataframe, config.input_size)
                    val_size = int(training_size * config.val_pct / (1 - config.test_pct - config.val_pct))
                    test_size = int(training_size * config.test_pct / (1 - config.test_pct - config.val_pct))
                    total_size_needed = training_size + val_size + test_size
                    # Ensure dataset is large enough or handle smaller datasets
                    dataset, _ = random_split(dataset, [total_size_needed, len(dataset) - total_size_needed])

                    train_dataset, val_dataset, test_dataset = random_split(dataset, [training_size, val_size, test_size])
                    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=num_workers, pin_memory=True)
                    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
                    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

                    # Instantiate the model based on the net type
                    model_class = net_classes[net_type]
                    model = model_class(config.input_size, config.hidden_size, config.output_size, config.dropout_prob).to(device)
                    optimizer = optim.Adam(model.parameters())
                    criterion = nn.MSELoss()
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
                    
                    total_training_time, avg_train_data_load_time, avg_train_process_time, avg_val_data_load_time, avg_val_process_time, test_loss = train_and_evaluate(model, train_loader, val_loader, test_loader, device, criterion, optimizer, scheduler, config, False)
                    predictions, actual = predict(model, test_loader, device)
                    mae, mse, rmse, r2 = calculate_metrics(predictions, actual)
                    
                    results.append({
                        "num_threads": num_threads,
                        "neural_net_type": net_type,  # Added attribute
                        "repetition": rep + 1,
                        "device": device_name,
                        "num_workers": num_workers,
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
                    current_test_duration = time.time() - start_test_time
                    total_elapsed_time = time.time() - start_all_tests_time
                    current_test += 1
                    print(f"Test {current_test} of {total_tests} on device: {device_name}, net_type: {net_type}, "
                          f"threads: {num_threads}, workers: {num_workers}, epochs: {epochs}, batch size: {batch_size}, "
                          f"training size: {training_size}, repetition: {rep + 1} ended after {current_test_duration:.2f} seconds.")
                    print(f"Total time elapsed since first test: {total_elapsed_time:.2f} seconds.")


    # Optionally, save final results with a different name to indicate completion
    final_results_df = pd.DataFrame(results)
    final_results_df.to_csv('experiment_results_final.csv', index=False)
    print("Final experiment results saved.")