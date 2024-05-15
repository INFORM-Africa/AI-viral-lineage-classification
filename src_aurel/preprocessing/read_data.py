import os, re, glob
import pandas as pd
from Bio import SeqIO
from .prepro import remove_ambiguous_bases


def extract_metadata(dataset_path):
    df_list = []
    for filepath in glob.glob(dataset_path + 'gisaid*.tsv'):
        df = pd.read_csv(filepath, sep='\t')
        meta = df[['Accession ID', 'Lineage', 'Collection date', 'Location']]
        df_list.append(meta)

    metadata_df = pd.concat(df_list, ignore_index=True)
    metadata_df['country'] = metadata_df['Location'].apply(lambda x: x.split("/")[1].strip())
    metadata_df.drop('Location', axis=1, inplace=True)

    dates_df = extract_dates(dataset_path)
    metadata_df = pd.merge(metadata_df, dates_df, on='Accession ID', how='inner')
    return metadata_df

def extract_dates(dataset_path):
    df_list = []
    for filepath in glob.glob(dataset_path + 'date*.tsv'):
        df = pd.read_csv(filepath, sep='\t')
        meta = df[['Accession ID', 'Submission date']]
        df_list.append(meta)

    dates_df = pd.concat(df_list, ignore_index=True)
    return dates_df

def clean_collection_dates(metadata_df, method='drop'):
    if method == 'drop':
        metadata_df['Collection date'] = pd.to_datetime(metadata_df['Collection date'], errors='coerce')
        metadata_df.dropna(subset=['Collection date'], inplace=True)
        metadata_df['date'] = metadata_df['Collection date']
    elif method == 'fill':
        metadata_df['date'] = metadata_df['Collection date']
        for index, row in metadata_df.iterrows():
            try:
                pd.to_datetime(row['date'])
            except ValueError:
                sub_date = row['Submission date']
                next_entry = metadata_df[(metadata_df['country'] == row['country']) & (metadata_df.index > index)].head(1)
                next_coll_date = next_entry['date'].values[0]
                metadata_df.at[index, 'date'] = min(sub_date, next_coll_date)
    else:
        raise ValueError("Invalid method. Choose between 'drop' and 'fill'.")

    return metadata_df

def extract_sequences(dataset_path):
    df_list = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".fasta"): 
            file_path = os.path.join(dataset_path, filename)
            for seq_record in SeqIO.parse(file_path, "fasta"):
                accession_id_match = re.search(r"EPI_ISL_\d+", seq_record.description)
                accession_id = accession_id_match.group(0) if accession_id_match else None
                df_list.append({"Accession ID": accession_id, "sequence": str(seq_record.seq)})

    sequences_df = pd.DataFrame(df_list)
    return sequences_df

def clean_sequences(sequences_df):
    sequences_df['sequence'] = sequences_df['sequence'].apply(remove_ambiguous_bases)
    return sequences_df

def merge_dfs(metadata_df=None, sequences_df=None, dataset_path=None):
    if metadata_df is None and sequences_df is None:
        if dataset_path is None:
            raise ValueError("Either metadata_df and sequences_df or dataset_path must be provided.")
        
    if metadata_df is None:
        metadata_df = extract_metadata(dataset_path)

    if sequences_df is None:
        sequences_df = extract_sequences(dataset_path)

    merged_df = pd.merge(metadata_df, sequences_df, on='Accession ID', how='inner')
    merged_df.rename(
        columns={'Lineage': 'lineage', 'Collection date' : 'col_date', 'Submission date': 'sub_date'}, 
        inplace=True
    )

    return merged_df

def remove_consensus_call_sequences(sequences_df):
    consensus_indexes = sequences_df[sequences_df['lineage'].str.contains('consensus call')].index
    sequences_df.drop(consensus_indexes, inplace=True)
    return sequences_df