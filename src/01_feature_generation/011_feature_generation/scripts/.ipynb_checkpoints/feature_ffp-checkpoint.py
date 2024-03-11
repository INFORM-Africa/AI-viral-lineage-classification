import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process DNA sequences using Chaos Game Representation (CGR).')
parser.add_argument('-f', '--File', type=str, required=True, help='Specify DSP Signal Feature File')
parser.add_argument('-v', '--Data', choices=['SARS', 'HIV'], required=True,
                        help='Specify the virus dataset.')
args = parser.parse_args()

print(f"Running FFP for {args.File}.")
data = pd.read_parquet(f'../../../data/features/{args.Data}/{args.File}.parquet', engine='pyarrow')  # You can use 'fastparquet' as the engine

targets = data["Target"]
train = data["Train"]

data = data.drop(columns = ["Target", "Train"])

data = data.apply(lambda x: x / x.sum(), axis=1)

data["Target"] = targets.to_list()
data["Train"] = train.to_list()

data.columns = data.columns.map(str)
output_filename = f'../../../data/features/{args.Data}/{args.File}_FFP.parquet'
data.to_parquet(output_filename, engine='pyarrow')
print(f"FFP features saved to {output_filename}")