import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def calculate_metrics(predictions, actual):
    # Convert lists of arrays into single numpy arrays
    predictions = np.vstack(predictions)
    actual = np.vstack(actual)
    
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)
    
    return mae, mse, rmse, r2