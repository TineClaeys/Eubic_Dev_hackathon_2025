import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
import re

def predict_entities(text, tokenizer, model, label_map, device):
    """
    Predicts named entities in the given text using a trained NER model.

    Args:
        text (str or list): The input text or list of tokens to predict entities from.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process the text.
        model (transformers.PreTrainedModel): Trained NER model.
        label_map (dict): Mapping of label names to IDs.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        list: A list of dictionaries containing predicted entities with their text, type, and token indices.
    """
    tokens = []
    is_split_into_words = False
    
    if isinstance(text, list):
        tokens = text
        is_split_into_words = True
    else:
        tokens = re.findall(r'\S+|\s+', text)
        tokens = [t for t in tokens if t.strip()]
    
    encoding = tokenizer(
        tokens if is_split_into_words else text,
        return_offsets_mapping=True,
        is_split_into_words=is_split_into_words,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    
    word_ids = encoding.word_ids()
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    predictions = predictions[0].cpu().numpy()
    idx2label = {v: k for k, v in label_map.items()}
    predicted_entities = []
    current_entity = None
    
    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        
        token = tokens[word_id] if is_split_into_words else text[encoding["offset_mapping"][0][i][0]:encoding["offset_mapping"][0][i][1]]
        label = idx2label.get(predictions[i], "O")
        
        if label.startswith("B-"):
            if current_entity:
                predicted_entities.append(current_entity)
            
            entity_type = label[2:]
            current_entity = {
                "text": token,
                "type": entity_type,
                "start_token": word_id,
                "end_token": word_id
            }
        elif label.startswith("I-") and current_entity and current_entity["type"] == label[2:]:
            current_entity["text"] += " " + token
            current_entity["end_token"] = word_id
        else:
            if current_entity:
                predicted_entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        predicted_entities.append(current_entity)
    
    return predicted_entities

def main():
    """
    Main function to predict named entities using a trained NER model.

    Parses command-line arguments to specify the model directory, input text or file, and output file.
    Loads the model and tokenizer, and predicts entities for the given input.
    """
    parser = argparse.ArgumentParser(description="Predict entities in a text using a trained NER model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model")
    parser.add_argument("--input_file", type=str, help="JSON file containing text to predict entities from")
    parser.add_argument("--output_file", type=str, default="predictions.json", help="File to save predictions to")
    parser.add_argument("--text", type=str, help="Text to predict entities from (alternative to input_file)")
    
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    
    with open(f"{args.model_dir}/label_map.json", "r") as f:
        label_map = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if args.text:
        entities = predict_entities(args.text, tokenizer, model, label_map, device)
        
        print("\nPredicted entities:")
        for entity in entities:
            print(f"{entity['text']} ({entity['type']})")
        
        with open(args.output_file, "w") as f:
            json.dump({"text": args.text, "entities": entities}, f, indent=2)
        
    elif args.input_file:
        with open(args.input_file, "r") as f:
            data = json.load(f)
        
        results = []
        
        if isinstance(data, list):
            for item in tqdm(data, desc="Predicting entities"):
                text = item.get("sentence", "")
                entities = predict_entities(text, tokenizer, model, label_map, device)
                results.append({"text": text, "entities": entities})
        else:
            for document in tqdm(data.get("documents", []), desc="Processing documents"):
                doc_results = {
                    "id": document.get("id", ""),
                    "sentences": []
                }
                
                for sentence in document.get("sentences", []):
                    text = sentence.get("text", "")
                    entities = predict_entities(text, tokenizer, model, label_map, device)
                    doc_results["sentences"].append({
                        "id": sentence.get("id", ""),
                        "text": text,
                        "entities": entities
                    })
                
                results.append(doc_results)
        
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"Predictions saved to {args.output_file}")
    else:
        print("Error: You must provide either --text or --input_file")

if __name__ == "__main__":
    main()