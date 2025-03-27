import os
import json
import re

def parse_ann_file(ann_file_path):
    """
    Parses a BRAT annotation (.ann) file and extracts entities, triggers, events, relations, and equivalences.

    Args:
        ann_file_path (str): Path to the .ann file.

    Returns:
        tuple: A tuple containing:
            - entities (dict): Extracted entities with their type, start, end, and text.
            - triggers (dict): Extracted triggers with their type, start, end, and text.
            - events (dict): Extracted events with their type, trigger, and arguments.
            - relations (dict): Extracted relations with their type, arg1, and arg2.
            - equivs (list): List of equivalence groups.
    """
    entities = {}
    triggers = {}
    events = {}
    relations = {}
    equivs = []

    with open(ann_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if not parts or len(parts) < 2:
                continue

            ann_id = parts[0]
            if ann_id.startswith('T'):  # Entities and Triggers
                type_offsets = parts[1].split(' ')
                ann_type = type_offsets[0]
                bgn = int(type_offsets[1])
                end = int(type_offsets[-1])
                text = parts[2] if len(parts) > 2 else ""
                if 'Trigger' in ann_type:  # Custom rule to classify triggers
                    triggers[ann_id] = {'type': ann_type, 'bgn': bgn, 'end': end, 'text': text}
                else:
                    entities[ann_id] = {'type': ann_type, 'bgn': bgn, 'end': end, 'text': text}

            elif ann_id.startswith('E'):  # Events
                event_parts = parts[1].split(' ')
                event_type, trigger = event_parts[0].split(':')
                arguments = {arg.split(':')[0]: arg.split(':')[1] for arg in event_parts[1:]}
                events[ann_id] = {'type': event_type, 'trigger': trigger, 'arguments': arguments}

            elif ann_id.startswith('R'):  # Relations
                relation_parts = parts[1].split(' ')
                rel_type = relation_parts[0]
                arg1 = relation_parts[1].split(':')[1]
                arg2 = relation_parts[2].split(':')[1]
                relations[ann_id] = {'type': rel_type, 'arg1': arg1, 'arg2': arg2}

            elif ann_id.startswith('*'):  # Equivs
                equiv_entities = parts[1].split(' ')[1:]
                equivs.append(equiv_entities)

    return entities, triggers, events, relations, equivs

def split_sentences(text):
    """
    Splits a given text into sentences based on punctuation.

    Args:
        text (str): The input text.

    Returns:
        list: A list of tuples, where each tuple contains the sentence text, start index, and end index.
    """
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[\.\?\!])\s')
    sentences = []
    start = 0
  
    for match in sentence_endings.finditer(text):
        end = match.start() + 1  # Include the period in the sentence
        sentence_text = text[start:end].strip()
        if sentence_text:  # Only add non-empty sentences
            sentences.append((sentence_text, start, end))
        start = end
    
    if start < len(text):
        last_sentence = text[start:].strip()
        if last_sentence:
            sentences.append((last_sentence, start, len(text)))
    
    if not sentences and text.strip():
        sentences.append((text.strip(), 0, len(text)))
    
    return sentences

def tag_iob(text, entities):
    """
    Tags tokens in the text with IOB (Inside-Outside-Beginning) format based on entities.

    Args:
        text (str): The input text.
        entities (dict): A dictionary of entities with their start, end, and type.

    Returns:
        list: A list of tuples, where each tuple contains a token and its IOB tag.
    """
    tokens = []
    token_spans = []
    for match in re.finditer(r'\S+', text):
        tokens.append(match.group())
        token_spans.append((match.start(), match.end()))
    
    tags = ['O'] * len(tokens)
    
    for entity_id, entity in entities.items():
        entity_bgn = entity['bgn']
        entity_end = entity['end']
        entity_type = entity['type']
        
        entity_tokens = []
        for i, (token_bgn, token_end) in enumerate(token_spans):
            if max(token_bgn, entity_bgn) < min(token_end, entity_end):
                entity_tokens.append(i)
        
        for i, token_idx in enumerate(entity_tokens):
            if i == 0:
                tags[token_idx] = f'B-{entity_type}'
            else:
                tags[token_idx] = f'I-{entity_type}'
    
    return list(zip(tokens, tags))

def convert_brat_to_json(input_folder, output_file_json, output_file_iob):
    """
    Converts BRAT annotation files (.txt and .ann) into JSON and IOB formats.

    Args:
        input_folder (str): Path to the folder containing .txt and .ann files.
        output_file_json (str): Path to save the JSON output.
        output_file_iob (str): Path to save the IOB JSON output.

    Returns:
        None
    """
    documents = []
    tagged_sentences = []
    file_pairs = {}
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            base_name = file_name[:-4]
            ann_file = f"{base_name}.ann"
            if os.path.exists(os.path.join(input_folder, ann_file)):
                file_pairs[base_name] = {
                    'txt': os.path.join(input_folder, file_name),
                    'ann': os.path.join(input_folder, ann_file)
                }
    
    for base_name, files in file_pairs.items():
        txt_file_path = files['txt']
        ann_file_path = files['ann']
        
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            entities, triggers, events, relations, equivs = parse_ann_file(ann_file_path)
            sentences = split_sentences(text)
            
            sentence_data = []
            for i, (sentence_text, sentence_bgn, sentence_end) in enumerate(sentences):
                sentence_entities = {}
                for ent_id, entity in entities.items():
                    if max(entity['bgn'], sentence_bgn) < min(entity['end'], sentence_end):
                        adjusted_entity = entity.copy()
                        adjusted_entity['bgn'] = max(0, entity['bgn'] - sentence_bgn)
                        adjusted_entity['end'] = min(entity['end'] - sentence_bgn, sentence_end - sentence_bgn)
                        sentence_entities[ent_id] = adjusted_entity
                
                iob_tags = tag_iob(sentence_text, sentence_entities)
                
                tagged_sentences.append({
                    'file_id': base_name,
                    'sentence_id': f's{i}',
                    'sentence': sentence_text,
                    'tokens': iob_tags
                })
                
                sentence_data.append({
                    'id': f's{i}',
                    'text': sentence_text,
                    'bgn': sentence_bgn,
                    'end': sentence_end
                })
            
            document = {
                'id': base_name,
                'metadata': {'input_file_address_ann': ann_file_path, 'input_file_address_txt': txt_file_path},
                'text': text,
                'entities': entities,
                'triggers': triggers,
                'events': events,
                'relations': relations,
                'equivs': equivs,
                'sentences': sentence_data
            }
            documents.append(document)
            print(f"Processed file pair: {base_name}")
        
        except Exception as e:
            print(f"Error processing file pair {base_name}: {str(e)}")
    
    output_data = {
        'metadata': {
            'num_documents': len(documents),
            'num_sentences': sum(len(doc['sentences']) for doc in documents)
        },
        'documents': documents
    }
    
    with open(output_file_json, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    
    with open(output_file_iob, 'w', encoding='utf-8') as iob_file:
        json.dump(tagged_sentences, iob_file, ensure_ascii=False, indent=4)
    
    print(f"Processed {len(documents)} documents with {len(tagged_sentences)} sentences.")
    print(f"JSON output written to {output_file_json}")
    print(f"IOB output written to {output_file_iob}")

if __name__ == "__main__":
    """
    Entry point for the script. Parses command-line arguments and converts BRAT files to JSON and IOB formats.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert BRAT annotation files to JSON and IOB format")
    parser.add_argument("--input_folder", required=True, help="Folder containing .txt and .ann files")
    parser.add_argument("--output_json", default="output.json", help="Output JSON file path")
    parser.add_argument("--output_iob", default="output_iob.json", help="Output IOB JSON file path")
    
    args = parser.parse_args()
    
    convert_brat_to_json(args.input_folder, args.output_json, args.output_iob)