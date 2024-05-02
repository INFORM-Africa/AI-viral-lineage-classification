import glob
import os
import pandas as pd
from Bio import SeqIO

def extract_metadata(directory_path):
    data_frames = []

    for filepath in glob.glob(directory_path + '*.tsv'):
        df = pd.read_csv(filepath, sep='\t')
        meta = df[['strain', 'pangolin_lineage', 'date', 'date_submitted']]
        data_frames.append(meta)

    all_metadata = pd.concat(data_frames, ignore_index=True)
    return all_metadata


def extract_sequences(directory_path):
    sequences_data = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".fasta") or filename.endswith(".fa"): 
            file_path = os.path.join(directory_path, filename)

            for seq_record in SeqIO.parse(file_path, "fasta"):
                sequences_data.append({
                    "strain": seq_record.id,
                    "sequence": str(seq_record.seq)
                })

    df = pd.DataFrame(sequences_data)
    return df

if __name__ == "__main__":
    metadata = extract_metadata('./code/data/')
    sequences = extract_sequences('./code/data/')
    dataset = pd.merge(metadata, sequences, on='strain', how='inner')
    dataset.to_parquet(f'./code/data/dataset.parquet', engine='pyarrow')