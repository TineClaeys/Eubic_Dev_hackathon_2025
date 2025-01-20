import os

output_file = "/home/compomics/git/proteomics-sample-metadata/template_column_names.txt"
processed_columns = set()
folder_path = "/home/compomics/git/proteomics-sample-metadata/templates"

for file_name in os.listdir(folder_path):
    if file_name.endswith(".sdrf.tsv"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            column_names = file.readline().strip().split("\t")
            for column_name in column_names:
                if column_name not in processed_columns:
                    processed_columns.add(column_name)
                    with open(output_file, "a") as output:
                        output.write(column_name + "\n")
