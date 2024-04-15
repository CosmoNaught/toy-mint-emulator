import pandas as pd

def save_results(results):
    """
    Save the current results to a CSV file.
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_results_temp.csv', index=False)
    print("Temporary experiment results saved.")