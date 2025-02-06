#native imports 
from os.path import dirname, abspath
import re
import json

#third party imports 
import pandas as pd




def read_entitiy_json()-> dict:
    # Open and read the JSON file
    
    pwd = dirname(dirname(abspath(__file__)))
    entity_path = pwd + "/files/entity_synonyms.json"
    with open(entity_path, 'r') as file:
        entity = json.load(file)
        return(entity)


def normalize_headers(headers:str) -> list[str]:
    """
    Normalize headers: lower case, strip whitespace, and remove punctuation.
    """
    normalized = []
    for header in headers:
        h = header.lower().strip()
        h = re.sub(r'[^\w\s]', '', h)  # remove punctuation
        normalized.append(h)
    return normalized


def find_key_for_value(found_entity: str, dict_entities: dict) -> str:
    for key, values in dict_entities.items():
        if found_entity in values:
            return key
    return None  # Return None if the value isn't found


def replace_key(d, old_key, new_key):
    if old_key in d:
        d[new_key] = d.pop(old_key)
        
        
def filter_md_columns(supp_data_df: pd.DataFrame, entity_dictionary: dict) -> dict:
    d_cols = [col.strip() for col in supp_data_df.columns]
    replace_keys = [x for val in entity_dictionary.values() for x in val]
    all_entity_terms = [x for x in entity_dictionary.keys()] + replace_keys
    # also keeps first column as this is where the sample IDs usually are
    keep_cns = [d_cols[0]] + [x for x in d_cols if x in all_entity_terms] + [x for x in d_cols if normalize_headers(x) in all_entity_terms]
    keep_cns = list(set(keep_cns))
    out_dict = {col: supp_data_df[col].tolist() for col in keep_cns if col in supp_data_df.columns}
    return(out_dict)


def replace_invalid_entities(filtered_supp: dict, entity_dictionary:dict):
    replace_keys = [x for val in entity_dictionary.values() for x in val]
    original_keys = list(filtered_supp.keys())
    for key in original_keys:
        if key in replace_keys:
            new_key = find_key_for_value(key, entity_dictionary)
            replace_key(filtered_supp, key, new_key)
        if normalize_headers(key) in replace_keys:
            new_key = find_key_for_value(normalize_headers(key), entity_dictionary)
            replace_key(filtered_supp, key, new_key)
    return(filtered_supp)


def match_entities(tables_list:list[pd.DataFrame])-> dict:
    print("Starting to find matching entities")
    # load information of fixed entities from json file 
    entity_synonyms = read_entitiy_json()
    for table in tables_list:
        metadata_filtered = filter_md_columns(table, entity_synonyms)
        metadata_final = replace_invalid_entities(metadata_filtered, entity_synonyms)
        return metadata_final














