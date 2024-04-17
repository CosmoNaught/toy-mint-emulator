import pandas as pd

def save_results(results):
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_results_temp.csv', index=False)
    print("Temporary experiment results saved.")