import pandas as pd

csv_file = "x:\\Cosmo\\projects\\toy-mint-emulator\\shared\\data\\mint_data_scaled.csv"
df = pd.read_csv(csv_file)

df.to_pickle("data.pkl")