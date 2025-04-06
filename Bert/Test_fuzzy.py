import re
import difflib
import os
import glob
from typing import List, Dict, Tuple, Any
import shutil
from tqdm import tqdm

def read_custom_ann_file(ann_file_path: str) -> Dict[str, List[str]]:
    """
    Reads entities from a custom .ann file and organizes them by entity type.

    Args:
        ann_file_path: Path to the .ann file.

    Returns:
        A dictionary where keys are entity types and values are lists of entity values.
    """
    entities_by_type = {}
    with open(ann_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                entity_type, entity_value = line.split(':', 1)
                entity_type = entity_type.strip()
                entity_value = entity_value.strip()
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity_value)
    return entities_by_type

def fuzzy_match_entities(entities: List[str], text: str, threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Matches entities in the given text using fuzzy matching and returns their coordinates.

    Args:
        entities: List of entity strings to search for.
        text: The text to search within.
        threshold: Minimum similarity score (0-1) to consider a match.

    Returns:
        A list of dictionaries containing match information.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    results = []
    sentence_positions = []
    current_pos = 0
    for sentence in sentences:
        sentence_positions.append(current_pos)
        current_pos += len(sentence) + 2
    for entity in entities:
        for sentence_idx, sentence in enumerate(sentences):
            words = sentence.split()
            for i in range(len(words)):
                for j in range(i + 1, min(i + 10, len(words) + 1)):
                    candidate = ' '.join(words[i:j])
                    similarity = difflib.SequenceMatcher(None, entity.lower(), candidate.lower()).ratio()
                    if similarity >= threshold:
                        preceding_words_in_sentence = ' '.join(words[:i])
                        if preceding_words_in_sentence:
                            preceding_words_in_sentence += ' '
                        start_pos = sentence_positions[sentence_idx] + len(preceding_words_in_sentence)
                        end_pos = start_pos + len(candidate)
                        results.append({
                            'entity': entity,
                            'match': candidate,
                            'similarity': round(similarity, 3),
                            'start': start_pos,
                            'end': end_pos,
                            'sentence': sentence
                        })
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results

def fuzzy_match_custom_ann_entities(ann_file_path: str, txt_file_path: str, threshold: float = 0.8) -> Dict[str, List[Dict[str, Any]]]:
    """
    Matches entities from a custom .ann file in the corresponding text file using fuzzy matching.

    Args:
        ann_file_path: Path to the custom .ann file.
        txt_file_path: Path to the text file.
        threshold: Minimum similarity score (0-1) to consider a match.

    Returns:
        A dictionary where keys are entity types and values are lists of match results.
    """
    entities_by_type = read_custom_ann_file(ann_file_path)
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    results_by_type = {}
    for entity_type, entity_values in entities_by_type.items():
        matches = fuzzy_match_entities(entity_values, text, threshold)
        for match in matches:
            match['entity_type'] = entity_type
        results_by_type[entity_type] = matches
    return results_by_type

def generate_brat_ann_file(matches_by_type: Dict[str, List[Dict[str, Any]]], output_file_path: str):
    """
    Generates a brat-compatible .ann file with the fuzzy match coordinates.

    Args:
        matches_by_type: Dictionary of entity types to their match results.
        output_file_path: Path for the output .ann file.
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        entity_id = 1
        for entity_type, matches in matches_by_type.items():
            for match in matches:
                line = f"T{entity_id}\t{entity_type} {match['start']} {match['end']}\t{match['match']}\n"
                f.write(line)
                entity_id += 1

def filter_best_matches(matches_by_type: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filters matches to keep only the best match for each entity.

    Args:
        matches_by_type: Dictionary of entity types to their match results.

    Returns:
        A filtered dictionary with only the best match for each entity.
    """
    filtered_matches = {}
    for entity_type, matches in matches_by_type.items():
        matches_by_entity = {}
        for match in matches:
            entity = match['entity']
            if entity not in matches_by_entity:
                matches_by_entity[entity] = []
            matches_by_entity[entity].append(match)
        best_matches = []
        for entity, entity_matches in matches_by_entity.items():
            entity_matches.sort(key=lambda x: x['similarity'], reverse=True)
            best_matches.append(entity_matches[0])
        filtered_matches[entity_type] = best_matches
    return filtered_matches

def export_to_csv(matches_by_type: Dict[str, List[Dict[str, Any]]], output_file_path: str):
    """
    Exports match results to a CSV file.

    Args:
        matches_by_type: Dictionary of entity types to their match results.
        output_file_path: Path for the output CSV file.
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("entity_type,entity,match,similarity,start,end,sentence\n")
        for entity_type, matches in matches_by_type.items():
            for match in matches:
                sentence = match['sentence'].replace('"', '""')
                f.write(f"{entity_type},\"{match['entity']}\",\"{match['match']}\",{match['similarity']},{match['start']},{match['end']},\"{sentence}\"\n")

def process_folder(input_folder: str, output_folder: str, threshold: float = 0.8):
    """
    Processes all .ann and .txt file pairs in a folder and performs fuzzy matching.

    Args:
        input_folder: Path to the folder containing .ann and .txt files.
        output_folder: Path to the output folder.
        threshold: Minimum similarity score for fuzzy matching.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    ann_files = glob.glob(os.path.join(input_folder, "*.ann"))
    for _, ann_file in tqdm(enumerate(ann_files)):
        basename = os.path.basename(ann_file).split('.')[0]
        txt_file = os.path.join(input_folder, f"{basename}.txt")
        if not os.path.exists(txt_file):
            print(f"Warning: No matching .txt file found for {ann_file}")
            continue
        print(f"Processing {basename}...")
        output_ann = os.path.join(output_folder, f"{basename}.ann")
        output_txt = os.path.join(output_folder, f"{basename}.txt")
        output_csv = os.path.join(output_folder, f"{basename}.csv")
        shutil.copy2(txt_file, output_txt)
        all_matches_by_type = fuzzy_match_custom_ann_entities(ann_file, txt_file, threshold)
        best_matches_by_type = filter_best_matches(all_matches_by_type)
        generate_brat_ann_file(best_matches_by_type, output_ann)
        export_to_csv(best_matches_by_type, output_csv)
    print(f"Processing complete. Results saved to {output_folder}")

def main():
    """
    Entry point for the script. Parses command-line arguments and processes the input folder.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Fuzzy Entity Matching for Folders of .ann and .txt Files')
    parser.add_argument('--input', type=str, required=True, help='Input folder containing .ann and .txt files')
    parser.add_argument('--output', type=str, required=True, help='Output folder for processed files')
    parser.add_argument('--threshold', type=float, default=0.8, help='Similarity threshold (0-1)')
    args = parser.parse_args()
    process_folder(args.input, args.output, args.threshold)

if __name__ == "__main__":
    main()