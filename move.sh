#!/bin/bash

# Create the destination folder if it doesn't exist
mkdir -p common_pmids/open_access

# Find all .txt files larger than 10 KB and move them along with their corresponding .ann files
find common_pmids -type f -name "*.txt" -size +10k | while read txt_file; do
    # Get the base name without extension
    base_name=$(basename "$txt_file" .txt)
    
    # Move the .txt file
    mv "$txt_file" common_pmids/open_access/
    
    # Move the corresponding .ann file if it exists
    ann_file="common_pmids/${base_name}.ann"
    if [ -f "$ann_file" ]; then
        mv "$ann_file" common_pmids/open_access/
    fi
done