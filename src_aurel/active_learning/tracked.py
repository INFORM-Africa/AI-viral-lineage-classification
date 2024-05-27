import pandas as pd


def get_tracked_lineages_df(lineages_df:pd.DataFrame):
    # Defining variants that are a lineage + descendants while excluding the descendant that are a separate variant
    # B.1.1.7 + descendants, B.1.351 + descendants, P.1 + descendants, B.1.617.2 + descendants, B.1.1.529 + descendants
    lineages = ["B.1.1.7", "B.1.351", "B.1.1.28.1", "B.1.617.2", "B.1.1.529"]
    exclude = ["B.1.1.529.2.86", "B.1.1.529.2.75.3.4.1.1.1.1", "B.1.1.529.2.74"]
    names = ["Alpha", "Beta", "Gamma", "Delta", "Omicron (Parents)"]
    dates = ["2020-12-29", "2020-12-29", "2020-12-29", "2021-06-15", "2021-11-26"]

    # Defining variants that are only a lineage
    # BA.2.86, CH.1.1, BA.2.74, B.1.429, B.1.525, B.1.526, B.1.617.1, B.1.617.3, P.2, B.1.621
    remaining_lineages = ["B.1.1.529.2.86", "B.1.1.529.2.75.3.4.1.1.1.1", "B.1.1.529.2.74", "B.1.429", "B.1.525", "B.1.526", "B.1.617.1", "B.1.617.3", "B.1.1.28.2", "B.1.621"]
    remaining_dates =  ["2023-09-01", "2023-09-01", "2023-09-01", "2021-02-26", "2021-02-26", "2021-02-26", "2021-05-07", "2021-05-07", "2021-02-26", "2021-09-21"]
    remaining_names = ["Omicron", "Omicron", "Omicron", "Epsilon", "Eta", "Iota", "Kappa", "N/A", "Zeta", "Mu"]

    variants_dict = {
        "lineage": [], 
        "who_label": [], 
        "designation_date": [], 
        "first_dataset_date": [], 
        "last_dataset_date": [], 
        "count": []
    }

    class_counts = lineages_df['full_lineage'].value_counts()

    # Getting the variants that are a lineage + descendants
    for lineage, name, date in zip(lineages, names, dates):
        parent_and_descendants_lineages_df = _get_lineage_and_descendants(lineages_df, lineage)
        parent_and_descendants_lineages = parent_and_descendants_lineages_df.full_lineage.unique()
        
        first_date = parent_and_descendants_lineages_df.date.min()
        last_date = parent_and_descendants_lineages_df.date.max()
        
        for descendant in parent_and_descendants_lineages:
            if descendant in exclude:
                continue
            variants_dict["lineage"].append(descendant)
            variants_dict["who_label"].append(name)
            variants_dict["designation_date"].append(date)
            variants_dict["first_dataset_date"].append(first_date)
            variants_dict["last_dataset_date"].append(last_date)
            variants_dict["count"].append(class_counts[descendant])
    
    # Getting the variants that are only a lineage
    for lineage, name, date in zip(remaining_lineages, remaining_names, remaining_dates):
        variants_dict["lineage"].append(lineage)
        variants_dict["who_label"].append(name)
        variants_dict["designation_date"].append(date)
        variants_dict["first_dataset_date"].append(lineages_df[lineages_df['full_lineage'] == lineage].date.min())
        variants_dict["last_dataset_date"].append(lineages_df[lineages_df['full_lineage'] == lineage].date.max())
        variants_dict["count"].append(class_counts[lineage])

    output_df = pd.DataFrame(variants_dict)
    output_df['designation_date'] = pd.to_datetime(output_df['designation_date'])
    
    return output_df

def _get_lineage_and_descendants(lineages_df:pd.DataFrame, target_lineage:str):
    descendants = lineages_df[lineages_df['full_lineage'].str.startswith(f"{target_lineage}.")]
    parent = lineages_df[lineages_df['full_lineage'] == target_lineage]
    return pd.concat([parent, descendants], axis=0)