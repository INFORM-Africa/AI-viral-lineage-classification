import pandas as pd

def main():
    
    data_path = '../../../data/sequences/SARS-CoV-2/training_set'
    df = pd.read_parquet(data_path, engine='pyarrow')

    random_sample = df.groupby('Target').sample(n=1, random_state=1)
    print(random_sample.shape)
    random_sample.to_parquet('distance_sample.parquet')

if __name__ == "__main__":
    main()
