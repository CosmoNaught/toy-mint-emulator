import time
import torch
from tqdm import tqdm

def train_and_evaluate(model, train_loader, val_loader, test_loader, device, criterion, optimizer, scheduler, config, use_tqdm=True):
    start_time = time.time()  # Start timing the training process

    train_data_loading_times = []
    train_processing_times = []
    val_data_loading_times = []
    val_processing_times = []

    # Use tqdm only if use_tqdm is True
    epochs_iterator = tqdm(range(config.epochs), desc='Training Progress') if use_tqdm else range(config.epochs)

    for epoch in epochs_iterator:
        model.train()
        train_loss = 0

        # Wrap the train_loader with tqdm conditionally
        train_loader_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} - Training Data Loading', leave=False) if use_tqdm else train_loader
        for data, target in train_loader_iterator:
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
        val_loader_iterator = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} - Validation Data Loading', leave=False) if use_tqdm else val_loader
        with torch.no_grad():
            for data, target in val_loader_iterator:
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
        test_loader_iterator = tqdm(test_loader, desc='Testing Data Loading', leave=False) if use_tqdm else test_loader
        for data, target in test_loader_iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    
    return total_training_time, avg_train_data_load_time, avg_train_process_time, avg_val_data_load_time, avg_val_process_time, test_loss
