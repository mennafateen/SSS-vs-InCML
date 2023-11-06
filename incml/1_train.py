import random
import os
import pathlib
import datetime

import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
from torch import nn
from transformers import AdamW

from utils import load_meta_learning_dataset
from utils import CollateWraper


def train_epoch(train_loader, model, loss_func, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress_bar = tqdm(train_loader, desc='Training', leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        inputs = {key: value.to(device) for key, value in batch['inputs'].items()}
        labels = batch["inputs"]["labels"].view(-1).to(device)

        outputs = model(**inputs)
        logits = outputs.logits

        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        total_correct += torch.sum(predictions == labels).item()
        total_samples += len(labels)
        progress_bar.set_postfix({'Epoch': epoch + 1, 'Accuracy': total_correct / total_samples})

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


if __name__ == '__main__':

    parent_folder = 'data/incml/'

    NUM_EXAMPLES = [0, 1, 3]
    TRUNC_LEN = 100
    BATCH_SIZE = 8

    random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Turn off cudnn benchmarking for deterministic behavior

    for eg in NUM_EXAMPLES:
        for q in range(1, 6):
            # load data
            print("Question: ", q)
            question_folder = os.path.join(parent_folder, "q{}".format(q))
            task_list = []
            with open(question_folder + "/tasks.json", "r") as f:
                try:
                    task_list = json.load(f)
                except FileNotFoundError:
                    print("train file not found")

            dataset = load_meta_learning_dataset(question_folder)

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            config = AutoConfig.from_pretrained('aubmindlab/bert-base-arabertv02', num_labels=6)
            tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02')

            model = AutoModelForSequenceClassification.from_pretrained('aubmindlab/bert-base-arabertv02',
                                                                       config=config).to(device)

            collate_fn_train = CollateWraper(tokenizer, dataset, eg, TRUNC_LEN, mode="train")

            trainset = dataset['train']
            train_loader = torch.utils.data.DataLoader(trainset, collate_fn=collate_fn_train, batch_size=BATCH_SIZE,
                                                       shuffle=True, drop_last=False)
            # define model and params

            num_epochs = 6
            progress_bar = tqdm(train_loader, desc='Training', leave=False)
            loss_func = nn.CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=1e-5)

            for epoch in tqdm(range(num_epochs)):
                avg_loss, accuracy = train_epoch(train_loader, model, loss_func, optimizer)
                tqdm.write(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            saved_models_dir = f"models/incml/q{q}/eg{eg}_{current_date}/"
            os.makedirs(saved_models_dir, exist_ok=True)

            model.save_pretrained(saved_models_dir)
            tokenizer.save_pretrained(saved_models_dir)

            print("Model and tokenizer saved to:", saved_models_dir)
