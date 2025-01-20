import os
import random
import shutil

# Paths and settings
base_path = "/home/compomics/git/Eubic_Dev_hackathon_2025/annotationday_2101"
annotators = ["Dirk", "Julian", "Lev", "Magnus", "Tine", "Veit"]
source_path = "/home/compomics/git/Eubic_Dev_hackathon_2025/common_pmids/open_access"  # Replace with the actual path to your text files
num_batches = 3
papers_per_batch = 5

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
                    
            # Run the script
if __name__ == "__main__":
    distribute_papers()
    print(f"Distributed papers to annotator folders in: {base_path}")
