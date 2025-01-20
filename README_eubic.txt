I have downloaded PubMed in tsv format from this link: https://a3s.fi/March-2024-PubMed/all_documents.tsv
This file is tab separated with the following columns:
    identifier
    authors
    journal
    year
    title
    text

First extract the unique PMIDs from the two files:
    169 pmids_github_eubic.list
    221 pmids_pride_eubic.list
    390 PMIDs in total
Then sort the files and compare:
    comm -12 pmids_pride_eubic.list pmids_github_eubic.list > common_pmids.list
    comm -23 pmids_pride_eubic.list pmids_github_eubic.list > unique_pmids_pride.list
    comm -13 pmids_pride_eubic.list pmids_github_eubic.list > unique_pmids_github.list
Add the files on puhti:
    /scratch/project_2001426/katerina/eubic/
Extract the corresponding lines from all_documents.tsv:
    awk -F'\t' 'NR==FNR {docs[$1]; next} $1 in docs' common_pmids.list all_documents.tsv > common_pmids.tsv
    awk -F'\t' 'NR==FNR {docs[$1]; next} $1 in docs' unique_pmids_pride.list all_documents.tsv > unique_pmids_pride.tsv
    awk -F'\t' 'NR==FNR {docs[$1]; next} $1 in docs' unique_pmids_github.list all_documents.tsv > unique_pmids_github.tsv
Sort the tsv files:
    sort -u common_pmids.tsv > tmp && mv tmp common_pmids.tsv
    sort -u unique_pmids_pride.tsv > tmp && mv tmp unique_pmids_pride.tsv
    sort -u unique_pmids_github.tsv > tmp && mv tmp unique_pmids_github.tsv
Generate some dummy (empty) matches files for the tagger2standoff script to run:
    touch common_pmids_matches.tsv unique_pmids_pride_matches.tsv unique_pmids_github_matches.tsv
Generate the txt files in a directory
    mkdir common_pmids unique_pmids_pride unique_pmids_github
    python3 tagger2standoff.py common_pmids.tsv common_pmids_matches.tsv common_pmids
    python3 tagger2standoff.py unique_pmids_pride.tsv unique_pmids_pride_matches.tsv unique_pmids_pride
    python3 tagger2standoff.py unique_pmids_github.tsv unique_pmids_github_matches.tsv unique_pmids_github
Tar directories:
    tar -czvf common_pmids.tar.gz -C common_pmids .
    tar -czvf unique_pmids_pride.tar.gz -C unique_pmids_pride .
    tar -czvf unique_pmids_github.tar.gz -C unique_pmids_github .

Note: There are currently 30 documents missing from the PRIDE list. This is because these were published after March 2024.
