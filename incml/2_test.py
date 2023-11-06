import os
import json
import pathlib
import datetime

import torch
from sklearn.metrics import cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from utils import pta, load_meta_learning_dataset, CollateWraper

TRUNC_LEN = 70
BATCH_SIZE = 8
parent_folder = "data/incml/"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def k_times(k):
    num_eg = [0, 1, 3]
    questions = {}
    for q in range(1, 6):
        # load data
        questions[q] = {}
        print("Question: ", q)

        for eg in num_eg:
            question_folder = os.path.join(parent_folder, "q{}".format(q))
            with open(question_folder + "/tasks.json", "r") as f:
                task_list = json.load(f)

            dataset = load_meta_learning_dataset(question_folder)

            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            saved_models_dir = f"models/incml/q{q}/eg{eg}_{current_date}/"

            tokenizer = AutoTokenizer.from_pretrained(saved_models_dir)

            model = AutoModelForSequenceClassification.from_pretrained(saved_models_dir).to(device)

            total_qwk_qs = 0
            total_qwk_as = 0

            total_pta_qs = [0, 0, 0, 0]
            total_pta_as = [0, 0, 0, 0]
            for j in range(k):
                collate_fn_test = CollateWraper(tokenizer, dataset, eg, TRUNC_LEN, mode="test")
                testsets = {}
                for task in task_list:
                    testsets[task] = dataset[task]['test']
                test_loaders = {}
                for task in task_list:
                    test_loaders[task] = torch.utils.data.DataLoader(testsets[task], collate_fn=collate_fn_test,
                                                                     batch_size=BATCH_SIZE,
                                                                     shuffle=False, drop_last=False)
                model.eval()

                with torch.no_grad():
                    all_unseen_ans_predictions = []
                    all_unseen_ques_predictions = []
                    all_unseen_ans_labels = []
                    all_unseen_ques_labels = []
                    for task, test_loader in test_loaders.items():

                        for batch in test_loader:

                            labels = batch["inputs"]["labels"].view(-1).to(device)
                            labels_cpu = labels.cpu().tolist()
                            if task == 'task1':
                                for label in labels_cpu:
                                    all_unseen_ques_labels.append(label)
                            else:
                                for label in labels_cpu:
                                    all_unseen_ans_labels.append(label)

                            batch = {k: v.to(device) for k, v in batch.items()}
                            outputs = model(**batch["inputs"])

                            logits = outputs.logits

                            predictions = torch.argmax(logits, dim=1)
                            if task == 'task1':
                                for pred in predictions.cpu().tolist():
                                    all_unseen_ques_predictions.append(pred)
                            else:
                                for pred in predictions.cpu().tolist():
                                    all_unseen_ans_predictions.append(pred)

                total_qwk_qs += cohen_kappa_score(all_unseen_ques_labels, all_unseen_ques_predictions,
                                                  weights='quadratic')
                test_pta_qs = pta(all_unseen_ques_labels, all_unseen_ques_predictions)

                total_qwk_as += cohen_kappa_score(all_unseen_ans_labels, all_unseen_ans_predictions,
                                                  weights='quadratic')
                test_pta_as = pta(all_unseen_ans_labels, all_unseen_ans_predictions)

                for i in range(4):
                    total_pta_qs[i] += test_pta_qs[i]
                    total_pta_as[i] += test_pta_as[i]

            avg_pta_qs = [test_pta_q / k for test_pta_q in total_pta_qs]
            avg_pta_as = [test_pta_a / k for test_pta_a in total_pta_as]

            avg_qwk_as = total_qwk_as / k
            avg_qwk_qs = total_qwk_qs / k

            questions[q][eg] = {"avg_qwk_ques": avg_qwk_qs, "avg_pta_ques": avg_pta_qs,
                                "avg_qwk_ans": avg_qwk_as, "avg_pta_ans": avg_pta_as}
        print(questions[q])


if __name__ == '__main__':
    k_times(10)
