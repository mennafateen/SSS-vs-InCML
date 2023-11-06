import os
import glob

from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Specify the directory containing CSV files
data_dir = 'data/classifier/train'

# List all CSV files in the directory
train_files = glob.glob(os.path.join(data_dir, '*.csv'))
test_file = 'data/classifier/test.csv'

for file in train_files:
    print("File: ", file)
    # Load the datasets
    train_dataset_dict = load_dataset('csv', data_files=file)

    # Access the 'train' split
    train_dataset = train_dataset_dict['train']

    test_dataset_dict = load_dataset('csv', data_files='data/classifier/test.csv')
    test_dataset = test_dataset_dict['train']

    MODEL_NAME = "aubmindlab/bert-base-arabertv02"
    model = SetFitModel.from_pretrained(MODEL_NAME)

    trainer = SetFitTrainer(
        model=model,
        seed=42,
        train_dataset=train_dataset,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=16,
        num_iterations=50,
        num_epochs=5,
        column_mapping={"text": "text", "label": "label"}
    )

    trainer.train()
    # model._save_pretrained("/models/classifier/aug3/")

    y_pred = model(test_dataset['text'])
    y_true = test_dataset['label']
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("F1: ", f1_score(y_true, y_pred, average='macro'))
    print("Precision: ", precision_score(y_true, y_pred, average='macro'))
    print("Recall: ", recall_score(y_true, y_pred, average='macro'))
    print("---------------------------------------------------")