import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import precision_score, recall_score, f1_score
import argparse
from tqdm import tqdm
import os
import logging
import random
from datetime import datetime
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ner_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = self._create_label_map()
        self.label_counts = Counter()
        logger.info(f"Created dataset with {len(data)} examples and {len(self.label_map)} labels")

    def _create_label_map(self):
        tags = set()
        for item in self.data:
            for _, tag in item["tokens"]:
                tags.add(tag)
        tags_list = sorted(tags)
        if "O" in tags_list:
            tags_list.remove("O")
            tags_list = ["O"] + tags_list
        return {tag: i for i, tag in enumerate(tags_list)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = [t[0] for t in item["tokens"]]
        tags = [t[1] for t in item["tokens"]]

        word_ids = []
        label_ids = []
        encoded_tokens = []

        for word_idx, (word, tag) in enumerate(zip(tokens, tags)):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.tokenizer.unk_token]
            encoded_tokens.extend(word_tokens)
            word_ids.extend([word_idx] * len(word_tokens))
            tag_id = self.label_map[tag]
            label_ids.extend([tag_id] * len(word_tokens))
            self.label_counts[tag_id] += 1  # count labels

        if len(encoded_tokens) > self.max_len - 2:
            encoded_tokens = encoded_tokens[: self.max_len - 2]
            word_ids = word_ids[: self.max_len - 2]
            label_ids = label_ids[: self.max_len - 2]

        encoded_tokens = [self.tokenizer.cls_token] + encoded_tokens + [self.tokenizer.sep_token]
        word_ids = [-100] + word_ids + [-100]
        label_ids = [-100] + label_ids + [-100]

        input_ids = self.tokenizer.convert_tokens_to_ids(encoded_tokens)
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        label_ids += [-100] * padding_length
        word_ids += [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(label_ids),
            "word_ids": torch.tensor(word_ids),
        }

    def get_label_map(self):
        return self.label_map

    def get_class_weights(self):
        counts = np.array([self.label_counts.get(i, 1) for i in range(len(self.label_map))])
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float)

def compute_metrics(p, idx2label):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []

    for pred, label in zip(preds, labels):
        true_seq = []
        pred_seq = []
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                true_seq.append(idx2label[l_])
                pred_seq.append(idx2label[p_])
        if true_seq:
            true_labels.append(true_seq)
            pred_labels.append(pred_seq)

    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    return {"precision": precision, "recall": recall, "f1": f1}

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

import torch.nn.functional as F

class HingeLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Flatten logits and labels
        logits = logits.view(-1, self.model.config.num_labels)  # (batch_size * seq_len, num_labels)
        labels = labels.view(-1)                                # (batch_size * seq_len)

        # Create active mask and apply it to both
        active_loss = labels != -100                            # Mask to ignore padding tokens
        active_logits = logits[active_loss]
        active_labels = labels[active_loss]

        # Convert labels to one-hot, then to -1/+1 for hinge loss
        labels_one_hot = F.one_hot(active_labels, num_classes=self.model.config.num_labels).float()
        binary_labels = 2 * labels_one_hot - 1  # Convert to -1 and 1

        # Hinge loss calculation (squared hinge)
        margins = 1 - active_logits * binary_labels
        hinge_loss = torch.mean(torch.clamp(margins, min=0) ** 2)

        return (hinge_loss, outputs) if return_outputs else hinge_loss



def main():
    parser = argparse.ArgumentParser(description="Train a BERT-based NER model with weighted loss")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dmis-lab/biobert-base-cased-v1.1")
    parser.add_argument("--output_dir", type=str, default="./ner_model")
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--train_split", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # entity_data = [item for item in data if any(tag != "O" for _, tag in item["tokens"])]
    # random.shuffle(entity_data)
    # data = entity_data
    train_size = int(len(data) * args.train_split)
    train_data, val_data = data[:train_size], data[train_size:]

    logger.info(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = NERDataset(train_data, tokenizer, max_len=args.max_len)
    val_dataset = NERDataset(val_data, tokenizer, max_len=args.max_len)
    label_map = train_dataset.get_label_map()
    class_weights = train_dataset.get_class_weights()
    idx2label = {v: k for k, v in label_map.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_map),
        id2label=idx2label,
        label2id=label_map,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_total_limit=1,
        logging_steps=10,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )


    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = HingeLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, idx2label)
    )
    # trainer = WeightedLossTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=DataCollatorForTokenClassification(tokenizer),
    #     compute_metrics=lambda p: compute_metrics(p, idx2label),
    #     class_weights=class_weights,
    # )
 
    trainer.train()
    eval_results = trainer.evaluate()

    logger.info("Final Evaluation:")
    logger.info(json.dumps(eval_results, indent=2))

    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    model.save_pretrained(os.path.join(output_dir, "final_model"))

    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=4)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    logger.info(f"Training complete. Model and outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
