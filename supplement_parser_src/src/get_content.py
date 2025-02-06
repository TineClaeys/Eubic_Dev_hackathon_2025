#third party imports
import pandas as pd
import docx


    

def get_word_tables(file_path:str)-> list[pd.DataFrame]:
    final_df_list=[]
    doc = docx.Document(file_path)
    tables = doc.tables
    if tables:
            all_tables = []
        # Iterate through each table in the document
            for table in tables:
                # Create a DataFrame structure with empty strings, sized by the number of rows and columns in the table
                df = [['' for _ in range(len(table.columns))] for _ in range(len(table.rows))]
                # Iterate through each row in the current table
                for i, row in enumerate(table.rows):
                    # Iterate through each cell in the current row
                    for j, cell in enumerate(row.cells):
                        # If the cell has text, store it in the corresponding DataFrame position
                        if cell.text:
                            df[i][j] = cell.text #take first line and put as column names
                # Convert the list of lists (df) to a pandas DataFrame and add it to the tables list
                
                all_tables.append(pd.DataFrame(df))
              
                # Print the list of DataFrames representing the tables
                for df in all_tables:
                #TODO should also add some check to see if the first line is text
                    df.columns = df.iloc[0] #take first line and put as column names
                    df, is_expression = find_table_start_from_df(df)
                    if is_expression:
                        final_df_list.append(df)
    return(final_df_list)      

def get_txt_tables(file_path:str)-> list[pd.DataFrame]:
    final_df_list = []
    with open(file_path, 'r', encoding='utf-8') as txt_file:
        #TODO get multiple tables
            data = txt_file.readlines()
            df = pd.DataFrame(data, columns=['Content'])
            df, is_expression = find_table_start_from_df(df)
            if is_expression:
                final_df_list.append(df)
    return(final_df_list)

def get_excel_tables(file_path:str):
    final_df_list=[]
    #TODO get multiple tables
    df = pd.read_excel(file_path)
    df, is_expression = find_table_start_from_df(df)
    if is_expression:
        final_df_list.append(df)
    return(final_df_list)
    
def get_csv_tables(file_path:str):
    final_df_list=[]
    #TODO get multiple tables
    df = pd.read_csv(file_path)
    df, is_expression = find_table_start_from_df(df)
    if is_expression:
        final_df_list.append(df)
    return(final_df_list)
        
def get_tsv_tables(file_path:str):
    final_df_list=[]
    #TODO get multiple tables
    df = pd.read_csv(file_path, delimiter='\t')
    df, is_expression = find_table_start_from_df(df)
    if is_expression:
        final_df_list.append(df)
    final_df_list.append(df)
    return(final_df_list)
        
def is_expression_table(df:pd.DataFrame)-> bool:
    """
    Determine if the extracted table is likely a protein expression table.
    Checks if any normalized column name matches common gene/protein terms.
    """
    gene_synonyms = {"gene", "genesymbol", "geneid", "protein", "proteinid", "proteinaccession", "uniprotkbaccession"}
    # Normalize gene synonyms: remove spaces for better matching
    normalized_gene_synonyms = {re.sub(r'[^\w\s]', '', term.lower().strip()).replace(" ", "") for term in gene_synonyms}
    # Normalize DataFrame column names
    normalized_cols = normalize_headers(df.columns)
    return any(col in normalized_gene_synonyms for col in normalized_cols)

def find_table_start_from_df(df: pd.DataFrame, nan_threshold=0.5):
    """
    Identifies where the table starts in a DataFrame by detecting the first structured row.
    Parameters:
    - df (DataFrame): The raw DataFrame (without headers).
    - nan_threshold (float, optional): The proportion of NaN values in a row to consider it as unstructured.
    Returns:
    - tuple: (DataFrame containing the extracted table, Boolean indicating if it's a protein expression table)
    """
    # Iterate over rows to detect the first structured row (header)
    table_start = None
    for index, row in df.iterrows():
        nan_ratio = row.isna().sum() / len(row)  # Compute NaN ratio
        # Detect the first row where less than 'nan_threshold' of the values are NaN
        if nan_ratio < nan_threshold:
            table_start = index
            break  # Found the table start, exit loop
    if table_start is None:
        raise ValueError("No structured table detected in the given DataFrame.")
    # Extract the table from the detected start row
    table_df = df.iloc[table_start:].reset_index(drop=True)
    # Set the first row as the header
    table_df.columns = table_df.iloc[0]
    table_df = table_df[1:].reset_index(drop=True)
    # Check if the extracted table is a protein expression table
    is_expression = is_expression_table(table_df)
    return(table_df, is_expression)