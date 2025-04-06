import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import argparse
from tqdm import tqdm
import os
import logging
import random
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ner_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NERDataset(Dataset):
    """
    A custom PyTorch Dataset for Named Entity Recognition (NER) tasks.

    Args:
        data (list): A list of dictionaries containing tokenized text and corresponding labels.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process the text.
        max_len (int): Maximum sequence length for tokenized inputs.

    Attributes:
        data (list): The input data.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance.
        max_len (int): Maximum sequence length.
        label_map (dict): Mapping of labels to integer IDs.
    """
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = self._create_label_map()
        logger.info(f"Created dataset with {len(data)} examples and {len(self.label_map)} labels")
        
    def _create_label_map(self):
        """
        Creates a mapping of labels to integer IDs.

        Returns:
            dict: A dictionary mapping labels to integer IDs.
        """
        tags = set()
        for item in self.data:
            for _, tag in item['tokens']:
                tags.add(tag)
        
        tags_list = sorted(list(tags))
        if 'O' in tags_list:
            tags_list.remove('O')
            tags_list = ['O'] + tags_list
            
        return {tag: i for i, tag in enumerate(tags_list)}
    
    def __len__(self):
        """
        Returns the number of examples in the dataset.

        Returns:
            int: Number of examples.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a single example from the dataset.

        Args:
            idx (int): Index of the example.

        Returns:
            dict: A dictionary containing input IDs, attention mask, labels, and word IDs.
        """
        item = self.data[idx]
        tokens = [t[0] for t in item['tokens']]
        tags = [t[1] for t in item['tokens']]
        
        word_ids = []
        label_ids = []
        encoded_tokens = []
        
        for word_idx, (word, tag) in enumerate(zip(tokens, tags)):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.tokenizer.unk_token]
                
            encoded_tokens.extend(word_tokens)
            word_ids.extend([word_idx] * len(word_tokens))
            label_ids.extend([self.label_map[tag]] * len(word_tokens))
        
        if len(encoded_tokens) > self.max_len - 2:  
            encoded_tokens = encoded_tokens[:self.max_len - 2]
            word_ids = word_ids[:self.max_len - 2]
            label_ids = label_ids[:self.max_len - 2]
        
        encoded_tokens = [self.tokenizer.cls_token] + encoded_tokens + [self.tokenizer.sep_token]
        word_ids = [-100] + word_ids + [-100] 
        label_ids = [-100] + label_ids + [-100]  
        
        input_ids = self.tokenizer.convert_tokens_to_ids(encoded_tokens)
        attention_mask = [1] * len(input_ids)
        
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            label_ids = label_ids + ([-100] * padding_length)
            word_ids = word_ids + ([-100] * padding_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'word_ids': torch.tensor(word_ids, dtype=torch.long)
        }
    
    def get_label_map(self):
        """
        Returns the label-to-ID mapping.

        Returns:
            dict: Label-to-ID mapping.
        """
        return self.label_map

def train_model(model, train_dataloader, val_dataloader, device, epochs=5, lr=2e-4, output_dir="./model"):
    """
    Trains the NER model.

    Args:
        model (transformers.PreTrainedModel): The model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to train the model on.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        output_dir (str): Directory to save the trained model.

    Returns:
        tuple: Trained model, training losses, validation losses, and best F1 score.
    """
    os.makedirs(output_dir, exist_ok=True)    
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    train_losses = []
    val_losses = []
    best_f1 = 0
    
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        model.train()
        epoch_loss = 0
        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_progress.set_postfix({"loss": loss.item()})
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        
        eval_results = evaluate_model(
            model=model,
            dataloader=val_dataloader,
            device=device,
            idx2label={v: k for k, v in train_dataloader.dataset.get_label_map().items()}
        )
        
        val_losses.append(eval_results['loss'])
        
        logger.info(f"Validation Loss: {eval_results['loss']:.4f}")
        logger.info(f"Validation Precision: {eval_results['precision']:.4f}")
        logger.info(f"Validation Recall: {eval_results['recall']:.4f}")
        logger.info(f"Validation F1: {eval_results['f1']:.4f}")
        
        if eval_results['f1'] > best_f1:
            best_f1 = eval_results['f1']
            logger.info(f"New best F1 score: {best_f1:.4f}, saving model")
            model_path = os.path.join(output_dir, "best_model")
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
            logger.info(f"Model saved to {model_path}")
    
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    return model, train_losses, val_losses, best_f1

def evaluate_model(model, dataloader, device, idx2label):
    """
    Evaluates the NER model on the validation dataset.

    Args:
        model (transformers.PreTrainedModel): The model to evaluate.
        dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to evaluate the model on.
        idx2label (dict): Mapping of label IDs to label names.

    Returns:
        dict: Evaluation metrics including loss, precision, recall, F1 score, and classification report.
    """
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            word_ids = batch['word_ids']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2)
            
            batch_size = preds.shape[0]
            for i in range(batch_size):
                pred_entities = []
                true_entities = []
                
                prev_word_idx = None
                for j, word_idx in enumerate(word_ids[i]):
                    word_idx = word_idx.item()
                    
                    if word_idx == -100 or word_idx == prev_word_idx:
                        continue
             
                    pred_label_id = preds[i, j].item()
                    true_label_id = labels[i, j].item()
                    
                    pred_label = idx2label.get(pred_label_id, 'O')
                    true_label = idx2label.get(true_label_id, 'O') if true_label_id != -100 else 'O'
                    
                    pred_entities.append(pred_label)
                    true_entities.append(true_label)
                    
                    prev_word_idx = word_idx
                
                if pred_entities:  
                    predictions.append(pred_entities)
                    true_labels.append(true_entities)
    
    avg_loss = total_loss / len(dataloader)
    
    if not predictions:
        return {
            'loss': avg_loss,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'report': "No predictions"
        }
    
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    return {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }

def main():
    """
    Main function to train and evaluate a BERT-based NER model.
    """
    parser = argparse.ArgumentParser(description='Train a BERT-based NER model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the IOB formatted JSON file')
    parser.add_argument('--model_name', type=str, default='allenai/scibert_scivocab_uncased', help='Base model to use')
    parser.add_argument('--output_dir', type=str, default='./ner_model', help='Directory to save the model')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.7, help='Train/val split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from {args.input_file}")
    
    random.shuffle(data)
    train_size = int(len(data) * args.train_split)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    logger.info(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")
    
    logger.info(f"Loading model {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = NERDataset(train_data, tokenizer, max_len=args.max_len)
    val_dataset = NERDataset(val_data, tokenizer, max_len=args.max_len)
    
    label_map = train_dataset.get_label_map()
    num_labels = len(label_map)
    idx2label = {v: k for k, v in label_map.items()}
    
    logger.info(f"Number of labels: {num_labels}")
    logger.info(f"Label map: {label_map}")
    
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=4)
    
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels,
        id2label=idx2label,
        label2id=label_map
    )
    model.to(device)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    
    trained_model, train_losses, val_losses, best_f1 = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
        output_dir=output_dir
    )
    
    logger.info("Performing final evaluation")
    eval_results = evaluate_model(
        model=trained_model,
        dataloader=val_dataloader,
        device=device,
        idx2label=idx2label
    )
    
    logger.info("\nFinal Evaluation Metrics:")
    logger.info(f"Precision: {eval_results['precision']:.4f}")
    logger.info(f"Recall: {eval_results['recall']:.4f}")
    logger.info(f"F1 Score: {eval_results['f1']:.4f}")
    logger.info(f"Classification Report:\n{eval_results['report']}")
    
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    logger.info(f"Tokenizer saved to {os.path.join(output_dir, 'final_model')}")
  
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_f1": best_f1,
            "final_metrics": {
                "precision": eval_results['precision'],
                "recall": eval_results['recall'],
                "f1": eval_results['f1']
            }
        }, f, indent=4)
    
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(eval_results['report'])
    
    logger.info(f"Training complete. All outputs saved to {output_dir}")

if __name__ == "__main__":
    main()