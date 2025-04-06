import os
import glob
import csv
import re
from typing import Dict, List, Tuple, Set
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def read_custom_ann_file(ann_file_path: str) -> Dict[str, Set[str]]:
    """
    Reads entities from a custom .ann file and organizes them by entity type.

    Args:
        ann_file_path: Path to the .ann file.

    Returns:
        A dictionary where keys are entity types and values are sets of entity values.
    """
    entities_by_type = defaultdict(set)
    with open(ann_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                entity_type, entity_value = line.split(':', 1)
                entity_type = entity_type.strip()
                entity_value = entity_value.strip()
                entities_by_type[entity_type].add(entity_value)
    return entities_by_type

def read_brat_ann_file(ann_file_path: str) -> Dict[str, Set[str]]:
    """
    Reads entities from a brat .ann file and organizes them by entity type.

    Args:
        ann_file_path: Path to the .ann file.

    Returns:
        A dictionary where keys are entity types and values are sets of entity values.
    """
    entities_by_type = defaultdict(set)
    with open(ann_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('T'):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    type_span = parts[1].split(' ')
                    entity_type = type_span[0]
                    entity_text = parts[2]
                    entities_by_type[entity_type].add(entity_text)
    return entities_by_type

def read_csv_results(csv_file_path: str) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Reads fuzzy matching results from a CSV file.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        A dictionary where keys are entity types and values are sets of tuples containing
        the original entity and the matched text.
    """
    entities_by_type = defaultdict(set)
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entity_type = row['entity_type']
            original_entity = row['entity'].strip('"')
            matched_text = row['match'].strip('"')
            entities_by_type[entity_type].add((original_entity, matched_text))
    return entities_by_type

def evaluate_matches(
    original_entities: Dict[str, Set[str]],
    matched_entities: Dict[str, Set[Tuple[str, str]]],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the recall of fuzzy matching results against the original entities.

    Args:
        original_entities: Dictionary of original entities grouped by type.
        matched_entities: Dictionary of matched entities grouped by type.

    Returns:
        A dictionary containing evaluation metrics (original count, matched count, recall)
        for each entity type and overall.
    """
    evaluation = {}
    all_original_count = 0
    all_matched_count = 0
    for entity_type in set(original_entities.keys()) | set(matched_entities.keys()):
        original = original_entities.get(entity_type, set())
        matched = matched_entities.get(entity_type, set())
        original_count = len(original)
        matched_count = len({orig for orig, _ in matched})
        recall = matched_count / original_count if original_count > 0 else 0
        evaluation[entity_type] = {
            'original_count': original_count,
            'matched_count': matched_count,
            'recall': recall
        }
        all_original_count += original_count
        all_matched_count += matched_count
    overall_recall = all_matched_count / all_original_count if all_original_count > 0 else 0
    evaluation['OVERALL'] = {
        'original_count': all_original_count,
        'matched_count': all_matched_count,
        'recall': overall_recall
    }
    return evaluation

def evaluate_folder(
    original_folder: str,
    results_folder: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Evaluates fuzzy matching results for all files in a folder.

    Args:
        original_folder: Path to the folder containing original .ann files.
        results_folder: Path to the folder containing result files.

    Returns:
        A tuple containing a DataFrame of detailed results and a dictionary of aggregated evaluation metrics.
    """
    all_results = []
    aggregated_evaluation = defaultdict(lambda: {'original_count': 0, 'matched_count': 0})
    ann_files = glob.glob(os.path.join(original_folder, "*.ann"))
    for original_ann_file in ann_files:
        basename = os.path.basename(original_ann_file).split('.')[0]
        results_csv_file = os.path.join(results_folder, f"{basename}.csv")
        if not os.path.exists(results_csv_file):
            continue
        original_entities = read_custom_ann_file(original_ann_file)
        matched_entities = read_csv_results(results_csv_file)
        evaluation = evaluate_matches(original_entities, matched_entities)
        for entity_type, metrics in evaluation.items():
            if entity_type != 'OVERALL':
                result = {
                    'file': basename,
                    'entity_type': entity_type,
                    'original_count': metrics['original_count'],
                    'matched_count': metrics['matched_count'],
                    'recall': metrics['recall']
                }
                all_results.append(result)
                aggregated_evaluation[entity_type]['original_count'] += metrics['original_count']
                aggregated_evaluation[entity_type]['matched_count'] += metrics['matched_count']
    for entity_type, counts in aggregated_evaluation.items():
        if counts['original_count'] > 0:
            counts['recall'] = counts['matched_count'] / counts['original_count']
        else:
            counts['recall'] = 0
    total_original = sum(metrics['original_count'] for metrics in aggregated_evaluation.values())
    total_matched = sum(metrics['matched_count'] for metrics in aggregated_evaluation.values())
    aggregated_evaluation['OVERALL'] = {
        'original_count': total_original,
        'matched_count': total_matched,
        'recall': total_matched / total_original if total_original > 0 else 0
    }
    results_df = pd.DataFrame(all_results)
    return results_df, aggregated_evaluation

def create_evaluation_report(
    results_df: pd.DataFrame,
    aggregated_evaluation: Dict[str, Dict[str, float]],
    output_file: str
):
    """
    Creates a markdown report summarizing the evaluation results.

    Args:
        results_df: DataFrame containing detailed evaluation results.
        aggregated_evaluation: Dictionary of aggregated evaluation metrics.
        output_file: Path to the output markdown file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Fuzzy Matching Evaluation Report\n\n")
        overall = aggregated_evaluation['OVERALL']
        f.write("## Overall Statistics\n\n")
        f.write(f"Total entities in original annotations: {overall['original_count']}\n\n")
        f.write(f"Total entities matched: {overall['matched_count']}\n\n")
        f.write(f"Overall recall: {overall['recall']:.2%}\n\n")
        f.write("## Performance by Entity Type\n\n")
        f.write("| Entity Type | Original Count | Matched Count | Recall |\n")
        f.write("|-------------|----------------|---------------|--------|\n")
        sorted_types = sorted(
            [t for t in aggregated_evaluation.keys() if t != 'OVERALL'],
            key=lambda t: aggregated_evaluation[t]['original_count'],
            reverse=True
        )
        for entity_type in sorted_types:
            metrics = aggregated_evaluation[entity_type]
            f.write(f"| {entity_type} | {metrics['original_count']} | {metrics['matched_count']} | {metrics['recall']:.2%} |\n")
        if len(results_df['file'].unique()) > 1:
            f.write("\n## Performance by File\n\n")
            file_stats = results_df.groupby('file').agg({
                'original_count': 'sum',
                'matched_count': 'sum'
            }).reset_index()
            file_stats['recall'] = file_stats['matched_count'] / file_stats['original_count']
            file_stats = file_stats.sort_values('recall', ascending=False)
            f.write("| File | Original Count | Matched Count | Recall |\n")
            f.write("|------|----------------|---------------|--------|\n")
            for _, row in file_stats.iterrows():
                f.write(f"| {row['file']} | {row['original_count']} | {row['matched_count']} | {row['recall']:.2%} |\n")

def plot_evaluation_charts(
    aggregated_evaluation: Dict[str, Dict[str, float]],
    output_folder: str
):
    """
    Plots evaluation charts for entity counts and recall.

    Args:
        aggregated_evaluation: Dictionary of aggregated evaluation metrics.
        output_folder: Path to the folder where charts will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    entity_stats = {k: v for k, v in aggregated_evaluation.items() if k != 'OVERALL'}
    sorted_types = sorted(
        entity_stats.keys(),
        key=lambda t: entity_stats[t]['original_count'],
        reverse=True
    )
    top_types = sorted_types[:15]
    plt.figure(figsize=(12, 8))
    original_counts = [entity_stats[t]['original_count'] for t in top_types]
    matched_counts = [entity_stats[t]['matched_count'] for t in top_types]
    x = range(len(top_types))
    width = 0.35
    plt.bar([i - width/2 for i in x], original_counts, width, label='Original')
    plt.bar([i + width/2 for i in x], matched_counts, width, label='Matched')
    plt.ylabel('Count')
    plt.title('Entity Counts by Type')
    plt.xticks(x, top_types, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'entity_counts.png'))
    plt.close()
    plt.figure(figsize=(12, 8))
    recalls = [entity_stats[t]['recall'] for t in top_types]
    plt.bar(x, recalls, width)
    plt.ylabel('Recall')
    plt.title('Recall by Entity Type')
    plt.xticks(x, top_types, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'entity_recall.png'))
    plt.close()

def main():
    """
    Main function to evaluate fuzzy matching results and generate reports and charts.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Fuzzy Entity Matching Results')
    parser.add_argument('--original', type=str, required=True, help='Folder with original .ann files')
    parser.add_argument('--results', type=str, required=True, help='Folder with result files')
    parser.add_argument('--output', type=str, default='evaluation', help='Output folder for evaluation results')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    results_df, aggregated_evaluation = evaluate_folder(args.original, args.results)
    output_report = os.path.join(args.output, 'evaluation_report.md')
    create_evaluation_report(results_df, aggregated_evaluation, output_report)
    charts_folder = os.path.join(args.output, 'charts')
    plot_evaluation_charts(aggregated_evaluation, charts_folder)
    output_csv = os.path.join(args.output, 'detailed_results.csv')
    results_df.to_csv(output_csv, index=False)
    overall = aggregated_evaluation['OVERALL']
    print(f"Total entities in original annotations: {overall['original_count']}")
    print(f"Total entities matched: {overall['matched_count']}")
    print(f"Overall recall: {overall['recall']:.2%}")

if __name__ == "__main__":
    main()