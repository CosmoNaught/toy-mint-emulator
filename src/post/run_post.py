import pandas as pd
import numpy as np
from tabulate import tabulate

def main():
    df = pd.read_csv("experiment_results_final.csv")
    print("Data Loaded")
    print(tabulate(df, headers='keys', tablefmt='psql'))