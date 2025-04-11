import os
import glob
import itertools
import numpy as np
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score, f1_score, jaccard_score
from fuzzywuzzy import fuzz

ANNOTATION_DIR = "/home/compomics/git/Eubic_Dev_hackathon_2025/Final_annotation/*/batch*/*.ann"

def parse_annotations(ann_file, with_coordinates=True):
    entities = set()
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if not parts[0].startswith('T') or len(parts) < 3:
                continue
            tag, span, text = parts
            fields = span.split()
            label = fields[0]
            if with_coordinates:
                coords = (fields[1], fields[-1])
                entities.add((label, text.lower().strip(), coords))
            else:
                entities.add((label, text.lower().strip()))
    return entities

def overlap(text1, text2):
    return fuzz.partial_ratio(text1, text2) > 95

def calculate_overlap_score(entities1, entities2):
    overlap_count = 0
    total_count = 0
    for entity1 in entities1:
        for entity2 in entities2:
            if overlap(entity1[1], entity2[1]):
                overlap_count += 1
            total_count += 1
    return overlap_count / total_count if total_count > 0 else 0

def calculate_pairwise_scores(annotations):
    annotators = list(annotations.keys())
    pairwise_scores = defaultdict(list)

    for a1, a2 in itertools.combinations(annotators, 2):
        ents1, ents2 = annotations[a1], annotations[a2]
        all_ents = list(ents1 | ents2)
        y1 = [1 if e in ents1 else 0 for e in all_ents]
        y2 = [1 if e in ents2 else 0 for e in all_ents]

        f1 = f1_score(y1, y2)
        kappa = cohen_kappa_score(y1, y2) if len(set(y1 + y2)) > 1 else float('nan')
        jaccard = jaccard_score(y1, y2)

        pairwise_scores[a1].append((f1, kappa, jaccard))
        pairwise_scores[a2].append((f1, kappa, jaccard))

    return pairwise_scores

def summarize_scores(scores_dict):
    summary = {}
    for annotator, scores in scores_dict.items():
        f1s = [s[0] for s in scores]
        kappas = [s[1] for s in scores]
        jaccards = [s[2] for s in scores]
        summary[annotator] = (
            np.mean(f1s),
            np.nanmean(kappas),
            np.mean(jaccards)
        )
    return summary

def main():
    ann_files = glob.glob(ANNOTATION_DIR)
    annotations_by_doc_with = defaultdict(lambda: defaultdict(set))
    annotations_by_doc_without = defaultdict(lambda: defaultdict(set))

    for ann_file in ann_files:
        parts = ann_file.split(os.sep)
        annotator = parts[-3]
        doc_id = os.path.basename(ann_file)
        annotations_by_doc_with[doc_id][annotator] = parse_annotations(ann_file, with_coordinates=True)
        annotations_by_doc_without[doc_id][annotator] = parse_annotations(ann_file, with_coordinates=False)

    all_scores_with = defaultdict(list)
    all_scores_without = defaultdict(list)

    for doc_id, annots in annotations_by_doc_with.items():
        if len(annots) < 2:
            continue
        doc_scores = calculate_pairwise_scores(annots)
        for annotator, scores in doc_scores.items():
            all_scores_with[annotator].extend(scores)

    for doc_id, annots in annotations_by_doc_without.items():
        if len(annots) < 2:
            continue
        doc_scores = calculate_pairwise_scores(annots)
        for annotator, scores in doc_scores.items():
            all_scores_without[annotator].extend(scores)

    results_with = summarize_scores(all_scores_with)
    results_without = summarize_scores(all_scores_without)

    print("Inter-Annotator Agreement (With Coordinates):")
    for annotator, (f1, kappa, jaccard) in results_with.items():
        print(f"{annotator}: F1 = {f1:.3f}, Kappa = {kappa:.3f}, Jaccard = {jaccard:.3f}")

    print("\nInter-Annotator Agreement (Without Coordinates):")
    for annotator, (f1, kappa, jaccard) in results_without.items():
        print(f"{annotator}: F1 = {f1:.3f}, Kappa = {kappa:.3f}, Jaccard = {jaccard:.3f}")

if __name__ == "__main__":
    main()
