import pandas as pd
from numpy import median
file_path = 'C:/Users/435328/Documents/lesSDRF/mock_supplementary_data_multiple_tables.txt'

def read_txt(file_path:str):
    with open(file_path, 'r', encoding='utf-8') as txt_file:
        input = txt_file.readlines()
    input = [x.replace('\n','').strip() for x in input]

    # Get indices of separators
    sep_i = [i for i in range(0,len(input)) if input[i] == '']

    # Extract lines into different list based on separators
    if len(sep_i) > 0:
        ranges = []
        for i in range(0, len(sep_i)+1):
            if i == 0:
                ranges.append((0, sep_i[i]))
            elif i == (len(sep_i)):
                ranges.append((sep_i[i-1]+1,len(input)))
            else:
                ranges.append((sep_i[i-1]+1,sep_i[i]))
        dfs = []

        for r in ranges:
            df = [input[i] for i in range(r[0],r[1])]
            if(df != []): # remove empty dataframes (empty lines)
                dfs.append(df)
        
        return dfs
    else:
        return input

def create_outputs_txt(df_list:list):
    tbs = []
    txs = []

    for l in df_list:
        l_split = [x.split('\t') for x in l]
        l_lengths = [len(x) for x in l_split]
        textlines = []
        tablelines = []

        med_length = median(l_lengths)

        # remove lines that have different length than the median and store them elsewhere
        for i in range(0, len(l_split)):
            line = l_split[i]
            if len(line) != med_length or med_length == 1:
                textlines.append(line)
            else:
                tablelines.append(line)
                
        if(len(tablelines) > 0):
            df = pd.DataFrame(tablelines[1:len(tablelines)], columns = tablelines[0])
            tbs.append(df)
        
        if(len(textlines) > 0):
            textlines = ['\t'.join(x) for x in textlines]
            text = '\n'.join(textlines)
            txs.append(text)
    
    return({'tables': tbs,
            'texts': txs})
