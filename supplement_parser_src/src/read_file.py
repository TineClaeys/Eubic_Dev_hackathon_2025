#native imports 
import os

#third party imports
import pandas as pd
import docx

#own imports
import src.get_content as cont

def check_extension(file_path:str):
    """
    Reads a supplementary file based on its extension.
    Supports CSV, TSV, Excel, DOCX (tables), and TXT files.
    """
    print("Starting to read in the supplementary file")
    extension = os.path.splitext(file_path)[1].lower()
    match extension:
        case '.csv':
            csv_tables = cont.get_csv_tables(file_path)
            return(csv_tables,)
        case '.tsv':
            tsv_tables = cont.get_tsv_tables(file_path) 
            return(tsv_tables,)      
        case '.xls' | '.xlsx':
            excel_tables = cont.get_excel_tables(file_path)  
            return(excel_tables,)    
        case '.docx':
            word_tables = cont.get_word_tables(file_path)
            word_text = cont.get_word_text(file_path)
            return(word_tables, word_text)
        case _:
            print("No valid file format")
            raise ValueError(f"Unsupported file extension: {extension}")

def read_supplementary_file():
    tables, texts = check_extension()
    return(tables,texts)
    #TODO do something with the matched table information 
    