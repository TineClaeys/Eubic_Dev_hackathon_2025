import os
import random
import shutil

# Paths and settings
base_path = "/home/compomics/git/Eubic_Dev_hackathon_2025/hackathon"
annotators = ["Dirk", "Julian", "Lev", "Magnus", "Tine", "Veit", "Maike", "Samuel", "Marta", "Karolina", "Tobias", "Armin", "Natalia", "Hari", "Tim", "Ana"]
source_path = "/home/compomics/git/Eubic_Dev_hackathon_2025/Combined_all/open_access"  # Replace with the actual path to your text files
num_batches = 10
papers_per_batch = 5
overlapping_papers = 3

# Ensure the source path exists
if not os.path.exists(source_path):
    raise FileNotFoundError(f"Source path does not exist: {source_path}")

# Get all available text files in the source path
txt_files = [f for f in os.listdir(source_path) if f.endswith('.txt')]
if len(txt_files) < num_batches * papers_per_batch:
    raise ValueError("Not enough text files to distribute among batches.")

# Randomly split files into batches
random.shuffle(txt_files)
batches = [txt_files[i * papers_per_batch:(i + 1) * papers_per_batch] for i in range(num_batches)]

# Function to distribute papers consistently across annotators
def distribute_papers():
    for annotator in annotators:
        annotator_path = os.path.join(base_path, annotator)
        os.makedirs(annotator_path, exist_ok=True)
        
        for batch_num, batch_files in enumerate(batches, start=1):
            batch_path = os.path.join(annotator_path, f"batch{batch_num}")
            os.makedirs(batch_path, exist_ok=True)
            
            for file_name in batch_files:
                # Copy the .txt file
                source_txt = os.path.join(source_path, file_name)
                target_txt = os.path.join(batch_path, file_name)
                shutil.copy(source_txt, target_txt)
                
                # Copy the corresponding .ann file
                ann_file = file_name.replace('.txt', '.ann')
                source_ann = os.path.join(source_path, ann_file)
                target_ann = os.path.join(batch_path, ann_file)
                if os.path.exists(source_ann):
                    shutil.copy(source_ann, target_ann)

papers = [f for f in os.listdir(source_path)if f.endswith('.txt')]
guideline_paper_path1 = "/home/compomics/git/Eubic_Dev_hackathon_2025/hackathon/guidelines/Dirk/batch1"
guideline_paper_path2 = "/home/compomics/git/Eubic_Dev_hackathon_2025/hackathon/guidelines/Dirk/batch2"
guideline_papers_batch1 = [f for f in os.listdir(guideline_paper_path1) if f.endswith('.txt')]
guideline_papers_batch2 = [f for f in os.listdir(guideline_paper_path2) if f.endswith('.txt')]
all_guideline_papers = guideline_papers_batch1 + guideline_papers_batch2
papers = [f for f in papers if f not in all_guideline_papers] 
target_root = "/home/compomics/git/Eubic_Dev_hackathon_2025/hackathon/annotator_agreement"      

import os
import shutil
from itertools import cycle

def distribute_papers(files, annotators, source_path, target_root):
    batches = []
    num_papers = len(files)
    num_annotators = len(annotators)
    
    if num_papers < 5:
        raise ValueError("Not enough papers to create a full batch.")
    
    index = 0
    annotator_cycle = cycle(annotators)  # Cycle through annotators
    
    while index + 5 <= num_papers:
        batch = files[index:index + 5]
        annotator = next(annotator_cycle)
        batches.append((annotator, batch))
        index += 2  # Move forward by 2 to keep 3 overlapping
    
    # Store batches in folders
    for batch_num, batch in enumerate(batches, start=1):
        annotator, batch_files = batch
        annotator_path = os.path.join(target_root, annotator)
        os.makedirs(annotator_path, exist_ok=True)
        
        batch_path = os.path.join(annotator_path, f"batch{batch_num}")
        os.makedirs(batch_path, exist_ok=True)
            
        for file_name in batch_files:
            # Copy the .txt file
            source_txt = os.path.join(source_path, file_name)
            target_txt = os.path.join(batch_path, file_name)
            shutil.copy(source_txt, target_txt)
            
            # Copy the corresponding .ann file
            ann_file = file_name.replace('.txt', '.ann')
            source_ann = os.path.join(source_path, ann_file)
            target_ann = os.path.join(batch_path, ann_file)
            if os.path.exists(source_ann):
                shutil.copy(source_ann, target_ann)

# Example usage
if __name__ == "__main__":    
    distribute_papers(papers, annotators, source_path, target_root)






# if __name__ == "__main__":
#     distribute_papers()
#     print(f"Distributed papers to annotator folders in: {base_path}")
