from src.read_file import read_supplementary_file
from src.match_entities import match_entities

def main():
    PATH = #input test file
    supp_table_list = read_supplementary_file(PATH)
    found_entities = match_entities(supp_table_list)
    print(found_entities)
    

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Supplement_Parser',
                    description='Parses entity information from supplementary files',
                    epilog='sup :D')
    
    
    parser.add_argument('-input') 
    return parser



if __name__ == "__main__":
    main()
