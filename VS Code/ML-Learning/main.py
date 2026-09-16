from pathlib import Path
import pandas as pd

file = Path(__file__).parent / "iot_data.csv"
data = pd.read_csv(file)

#data.iloc[rows, columns]
print(data.iloc[:, 0])