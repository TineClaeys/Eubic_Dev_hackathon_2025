import os
import glob
import itertools
import numpy as np
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt

ANNOTATION_DIR = "/home/compomics/git/Eubic_Dev_hackathon_2025/Final_annotation/*/batch*/*.ann"

def parse_ann_file(filepath):
    """Parses a BRAT .ann file and returns a set of (start, end, type, text)."""
    annotations = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith("T"):
                continue
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            eid, meta, text = parts
            tokens = meta.split()
            label = tokens[0]
            try:
                start = int(tokens[1])
                end = int(tokens[-1])
                annotations.add((start, end, label, text.strip().lower()))
            except ValueError:
                continue
    return annotations

def compute_entity_type_agreement(all_docs):
    """Compute F1 and Kappa for each entity type across documents."""
    f1_by_type = defaultdict(list)
    kappa_by_type = defaultdict(list)

    for doc_id, annotator_data in all_docs.items():
        for (ann1, ann2) in itertools.combinations(annotator_data.keys(), 2):
            ents1 = annotator_data[ann1]
            ents2 = annotator_data[ann2]

            all_spans = sorted(set([(s, e, t) for (s, e, t, txt) in ents1] + [(s, e, t) for (s, e, t, txt) in ents2]))

            for entity_type in set(t for (_, _, t) in all_spans):
                y_true = []
                y_pred = []

                for span in all_spans:
                    if span[2] != entity_type:
                        continue
                    y_true.append(1 if span in [(s, e, t) for (s, e, t, txt) in ents1] else 0)
                    y_pred.append(1 if span in [(s, e, t) for (s, e, t, txt) in ents2] else 0)

                if len(set(y_true + y_pred)) > 1:
                    f1 = f1_score(y_true, y_pred)
                    kappa = cohen_kappa_score(y_true, y_pred)
                    f1_by_type[entity_type].append(f1)
                    kappa_by_type[entity_type].append(kappa)

    avg_f1 = {k: np.mean(v) for k, v in f1_by_type.items()}
    avg_kappa = {k: np.mean(v) for k, v in kappa_by_type.items()}
    return avg_f1, avg_kappa

def plot_scores(scores_dict, title, ylabel, color):
    entity_types = list(scores_dict.keys())
    values = [scores_dict[e] for e in entity_types]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(entity_types, values, color=color)
    ax.set_xlabel('Entity Type')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(entity_types, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def main():
    all_docs = defaultdict(lambda: defaultdict(set))

    ann_files = glob.glob(ANNOTATION_DIR)
    for filepath in ann_files:
        parts = filepath.strip().split(os.sep)
        if len(parts) < 3:
            continue
        annotator = parts[-3]
        doc_id = parts[-1].replace(".ann", "")
        all_docs[doc_id][annotator] = parse_ann_file(filepath)

    avg_f1, avg_kappa = compute_entity_type_agreement(all_docs)

    plot_scores(avg_f1, "F1 Score per Entity Type", "F1 Score", color="skyblue")
    plot_scores(avg_kappa, "Cohen's Kappa per Entity Type", "Kappa Score", color="lightgreen")

    print("Average F1 Score per Entity Type:")
    for et, score in avg_f1.items():
        print(f"{et}: {score:.3f}")
    print("\nAverage Kappa Score per Entity Type:")
    for et, score in avg_kappa.items():
        print(f"{et}: {score:.3f}")

if __name__ == "__main__":
    main()
