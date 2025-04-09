import pandas as pd
import json
import numpy as np

# Step 1: Preprocess step to merge duplicate columns with the same base name
def merge_duplicate_columns(df):
    """
    Merges columns with the same base name (e.g., characteristics[organism] and characteristics[organism].1)
    into a single column, separated by commas, and drops the original columns.
    """
    base_columns = [i.split(']')[0] for i in df.columns]
    duplicate_base_columns = list(set([i for i in base_columns if base_columns.count(i) > 1]))

    for c in duplicate_base_columns:
        columns = [i for i in df.columns if c in i]
        # Merge columns and drop original columns
        df[f'new_{c}'] = df[columns].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)
        df = df.drop(columns, axis=1)
        # Rename the new column
        df = df.rename(columns={f'new_{c}': f'{c}]'})
    
    return df

# Step 2: Extract samples and biological replicate information
def extract_samples(sdrf):
    samples = {}
    sample_names = sdrf['source name'].unique()
    for idx, sample_name in enumerate(sample_names):
        sample_data = sdrf[sdrf['source name'] == sample_name].iloc[0]
        samples[f"sample {idx + 1}"] = {
            "name": sample_name,
            "biological_replicate": sample_data.get('characteristics[biological replicate]', '1')
        }
    return samples

# Step 3: Extract runs and assays from the data, considering labels
def extract_runs_and_assays(sdrf):
    runs = {}
    run_index = 1

    for idx, row in sdrf.iterrows():
        data_file = row['comment[data file]']

        # Check if the data file already exists in the runs dictionary
        run_name = None
        for existing_run_name, run_data in runs.items():
            if run_data['data file'] == data_file:
                run_name = existing_run_name
                break
        if not run_name:
            # New run
            run_name = f"run {run_index}"
            runs[run_name] = {
                "assays": [],
                "file uri": f"ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2018/08/{row['comment[data file]']}",
                "data file": row['comment[data file]']
            }
            run_index += 1
        
        # Extract sample and assay details
        sample_name = row['source name']
        fraction = row.get('comment[fraction]', '1')
        technical_replicate = row.get('comment[technical replicate]', '1')

        # Construct the factor values (e.g., for enrichment method, time, etc.)
        factor_values = {}
        for column in row.index:
            if 'factor value' in column:
                factor_name = column.split('[', 1)[1].split(']', 1)[0]
                factor_values[factor_name] = row[column]

        # Extract the label from the comment column
        label = row.get('comment[label]', None)

        # If the label exists, create an assay for this label
        if label:
            assay = {
                "sample": sample_name,
                "label": label,
                "fraction": fraction,
                "technical replicate": technical_replicate,
                "factor values": factor_values
            }
            runs[run_name]["assays"].append(assay)

    return runs

# Step 4: Extract factor values for different labelling channels
def extract_factor_values(sdrf):
    factor_columns = [column for column in sdrf.columns if 'factor value' in column]
    factor_values = {}
    for column in factor_columns:
        name = column.split('[', 1)[1].split(']', 1)[0]
        factor_values[name] = sdrf[column].unique()
    return factor_values

# Step 5: Extract sample characteristics
def extract_sample_characteristics(sdrf):
    characteristic_columns = [col for col in sdrf.columns if 'characteristics' in col]
    leave_out = ['biological replicate']
    for column in sdrf.columns[sdrf.columns.str.contains('factor value')]:
        name = column.split('[', 1)[1].split(']', 1)[0]
        leave_out.append(name)

    characteristic_columns = [characteristic for characteristic in characteristic_columns if not any(leave_out in characteristic for leave_out in leave_out)]
    
    # Create common and not shared groups for sample characteristics
    sample_characteristics = {}
    common_group = {"apply": "ALL"}
    shared_cols = []
    for col in characteristic_columns:
        unique_values = sdrf[col].unique()
        if len(unique_values) == 1:
            char_name = col.split('[')[1].split(']')[0]
            shared_cols.append(col)
            if char_name not in common_group:
                common_group[char_name] = unique_values[0]
    
    sample_characteristics["common"] = common_group

    notshared_characteristic_columns = [col for col in characteristic_columns if col not in shared_cols]
    grouped_samples = {}

    if len(notshared_characteristic_columns) > 0:
        for idx, row in sdrf.iterrows():
            sample_values = tuple(row[col] for col in notshared_characteristic_columns)
            if sample_values not in grouped_samples:
                grouped_samples[sample_values] = []
            grouped_samples[sample_values].append(row['source name'])

        group_counter = 1
        for sample_values, samples_in_group in grouped_samples.items():
            group_key = f"sample_chara_{group_counter}"
            group_counter += 1
            group = {"apply": samples_in_group}
            for col, value in zip(notshared_characteristic_columns, sample_values):
                characteristic_name = col.split('[')[1].split(']')[0]
                group[characteristic_name] = value
            sample_characteristics[group_key] = group

    return sample_characteristics

# Step 6: Extract assay characteristics
def extract_assay_characteristics(sdrf):
    comment_columns = [col for col in sdrf.columns if 'comment' in col]
    leave_out = ['technical replicate', 'biological replicate', 'file uri', 'data file', 'fraction', 'label']
    
    for column in sdrf.columns[sdrf.columns.str.contains('factor value')]:
        name = column.split('[', 1)[1].split(']', 1)[0]
        leave_out.append(name)

    comment_columns = [comment for comment in comment_columns if not any(leave_out in comment for leave_out in leave_out)]
    assay_characteristics = {}
    common_group = {"apply": "ALL"}
    shared_cols = []
    
    for col in comment_columns:
        unique_values = sdrf[col].unique()
        if len(unique_values) == 1:
            char_name = col.split('[')[1].split(']')[0]
            shared_cols.append(col)
            if char_name not in common_group:
                common_group[char_name] = unique_values[0]
    
    assay_characteristics["common"] = common_group

    notshared_comment_columns = [col for col in comment_columns if col not in shared_cols]
    grouped_samples = {}

    if len(notshared_comment_columns) > 0:
        for idx, row in sdrf.iterrows():
            sample_values = tuple(row[col] for col in notshared_comment_columns)
            if sample_values not in grouped_samples:
                grouped_samples[sample_values] = []
            grouped_samples[sample_values].append(row['assay name'])

        group_counter = 1
        for sample_values, samples_in_group in grouped_samples.items():
            group_key = f"assay_chara_{group_counter}"
            group_counter += 1
            group = {"apply": samples_in_group}
            for col, value in zip(notshared_comment_columns, sample_values):
                characteristic_name = col.split('[')[1].split(']')[0]
                group[characteristic_name] = value
            assay_characteristics[group_key] = group

    return assay_characteristics

# Final function to combine everything
def convert_sdrf(sdrf_basis):
    # Preprocess step: Merge duplicate columns
    sdrf = merge_duplicate_columns(sdrf_basis.copy())
    
    # Extract samples
    samples = extract_samples(sdrf)
    
    # Extract runs and assays
    runs = extract_runs_and_assays(sdrf)
    
    # Extract factor values
    factor_values = extract_factor_values(sdrf)
    
    # Extract sample characteristics
    sample_characteristics = extract_sample_characteristics(sdrf)
    
    # Extract assay characteristics
    assay_characteristics = extract_assay_characteristics(sdrf)
    
    # Prepare the final JSON object
    json_data = {
        "samples": samples,
        "runs": runs,
        "factor values": factor_values,
        "sample characteristics": sample_characteristics,
        "assay characteristics": assay_characteristics
    }

    # Convert int64 and numpy arrays to lists
    def convert_data(data):
        if isinstance(data, dict):
            return {key: convert_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [convert_data(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.int64):
            return int(data)
        return data
    
    json_data = convert_data(json_data)
    json_string = json.dumps(json_data, indent=4)
    return json_string

df = pd.read_csv('/home/compomics/git/Eubic_Dev_hackathon_2025/sdrf-json/PXD017710-tmt.sdrf.tsv', sep='\t')
json_output = convert_sdrf(df)

