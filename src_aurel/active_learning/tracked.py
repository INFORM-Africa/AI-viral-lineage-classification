import pandas as pd

def get_earliest_dates(return_df:bool=True):
    dates = {
        "Alpha": "2020-01-19",
        "Beta": "2020-02-18",
        "Gamma": "2020-04-05",
        "Delta": "2020-03-24",
        "Eta": "2020-03-25",
        "Iota": "2020-01-28",
        "Kappa": "2020-03-03",
        "N/A": "2021-01-03",
        "Zeta": "2020-11-15",
        "Mu": "2020-09-19",
        "Epsilon": "2020-01-26",
        "Omicron-CH.1.1": "2022-02-10",
        "Omicron-BA.1": "2020-01-05",
        "Omicron-BA.2": "2020-03-28",
        "Omicron-BA.2.74": "2022-01-01",
        "Omicron-BA.2.86": "2023-07-28",
        "Omicron-BA.3": "2021-11-18",
        "Omicron-BA.4": "2020-07-01",
        "Omicron-BA.5": "2020-06-30",
    }
    if not return_df:
        return dates
    
    dates_dict = {
        "who_label": list(dates.keys()),
        "earliest_date": list(dates.values())
    }
    dates_df =  pd.DataFrame(dates_dict)
    dates_df['earliest_date'] = pd.to_datetime(dates_df['earliest_date'])
    return dates_df


def get_tracked_lineages_df(lineages_df:pd.DataFrame):
    # Defining variants that are a lineage + descendants while excluding the descendant that are a separate variant
    # B.1.1.7 + descendants, B.1.351 + descendants, P.1 + descendants, B.1.617.2 + descendants, B.1.1.529 + descendants
    family_lineages_variants = ["B.1.1.7", "B.1.351", "B.1.1.28.1", "B.1.617.2", "B.1.1.529.2.75.3.4.1.1.1.1", "B.1.1.529.1", "B.1.1.529.2", "B.1.1.529.2.74", "B.1.1.529.2.86", "B.1.1.529.3", "B.1.1.529.4", "B.1.1.529.5"]
    family_lineages_names = ["Alpha", "Beta", "Gamma", "Delta", "Omicron-CH.1.1", "Omicron-BA.1", "Omicron-BA.2", "Omicron-BA.2.74", "Omicron-BA.2.86", "Omicron-BA.3", "Omicron-BA.4", "Omicron-BA.5"]
    family_lineages_dds = ["2020-12-29", "2020-12-29", "2020-12-29", "2021-06-15", "2023-09-01", "2021-11-26", "2021-11-26", "2023-09-01", "2023-09-01", "2021-11-26", "2021-11-26", "2021-11-26"]

    # Defining variants that are only a lineage
    # B.1.429, B.1.525, B.1.526, B.1.617.1, B.1.617.3, P.2, B.1.621
    single_lineage_variants = ["B.1.429", "B.1.525", "B.1.526", "B.1.617.1", "B.1.617.3", "B.1.1.28.2", "B.1.621"]
    single_lineage_names = ["Epsilon", "Eta", "Iota", "Kappa", "N/A", "Zeta", "Mu"]
    single_lineage_dds =  ["2021-02-26", "2021-02-26", "2021-02-26", "2021-05-07", "2021-05-07", "2021-02-26", "2021-09-21"]

    variants_dict = {
        "lineage": [], 
        "who_label": [], 
        "earliest_date": [], 
        "designation_date": [], 
        "first_dataset_date": [], 
        "last_dataset_date": [], 
        "count": []
    }
    earliest_dates = get_earliest_dates(return_df=False)
    class_counts = lineages_df.full_lineage.value_counts()

    # Getting the variants that are a lineage + descendants
    for lineage, name, date in zip(family_lineages_variants, family_lineages_names, family_lineages_dds):
        parent_and_descendants_lineages_df = _get_lineage_and_descendants(lineages_df, lineage)
        parent_and_descendants_lineages = parent_and_descendants_lineages_df.full_lineage.unique()
        
        first_date = parent_and_descendants_lineages_df.date.min()
        last_date = parent_and_descendants_lineages_df.date.max()
        earliest_date = earliest_dates[name]
        
        for descendant in parent_and_descendants_lineages:
            variants_dict["lineage"].append(descendant)
            variants_dict["who_label"].append(name)
            variants_dict["earliest_date"].append(earliest_date)
            variants_dict["designation_date"].append(date)
            variants_dict["first_dataset_date"].append(first_date)
            variants_dict["last_dataset_date"].append(last_date)
            variants_dict["count"].append(class_counts[descendant])
    
    # Getting the variants that are only a lineage
    for lineage, name, date in zip(single_lineage_variants, single_lineage_names, single_lineage_dds):
        variants_dict["lineage"].append(lineage)
        variants_dict["who_label"].append(name)
        variants_dict["designation_date"].append(date)
        variants_dict["earliest_date"].append(earliest_dates[name])
        variants_dict["first_dataset_date"].append(lineages_df[lineages_df.full_lineage == lineage].date.min())
        variants_dict["last_dataset_date"].append(lineages_df[lineages_df.full_lineage == lineage].date.max())
        variants_dict["count"].append(class_counts[lineage])

    output_df = pd.DataFrame(variants_dict)
    output_df['designation_date'] = pd.to_datetime(output_df['designation_date'])
    output_df['earliest_date'] = pd.to_datetime(output_df['earliest_date'])
    
    return output_df

def _get_lineage_and_descendants(lineages_df:pd.DataFrame, target_lineage:str):
    if target_lineage == "BA.2":
        descendants = lineages_df[(lineages_df.full_lineage.str.startswith("BA.2.")) & (~lineages_df.lineage.str.startswith("BA.2.86")) & (~lineages_df.lineage.str.startswith("BA.2.74"))]
    else:
        descendants = lineages_df[lineages_df.full_lineage.str.startswith(f"{target_lineage}.")]

    parent = lineages_df[lineages_df.full_lineage == target_lineage]
    return pd.concat([parent, descendants], axis=0)

