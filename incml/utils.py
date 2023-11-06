from collections import defaultdict
import os
import json
import random

import numpy as np
import torch



def pta(y_true, y_pred):
    """
    pta stands for percentage by tick accuracy and is a metric that is used to evaluate the performance of the model.
    # pta0 is the percentage of examples where the model's prediction is within 0 ticks of the true class, which is the
    # same as the accuracy. pta1 is the percentage of examples where the model's prediction is within 1 tick of the true
    # class (i.e. the model is off by 1 tick) and so on.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: pta0, pta1, pta2, pta3
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ticks = np.abs(y_true - y_pred)
    pta = np.zeros(4)
    for i in range(4):
        pta[i] = np.sum(ticks <= i) / len(y_true)
    return pta

def compute_distribution(output):
    append_keys = {}

    for k, v in output.items():
        dist, new_key, total_count = defaultdict(float), k + '_dist', 0.
        for d in v:
            n_rating = (d['gr'] >= 0)
            total_count += n_rating
            if (d['gr'] != -1):
                dist[d['gr']] += 1. / n_rating
        for l in dist:
            dist[l] /= total_count
        append_keys[new_key] = dist

    for k in append_keys:
        output[k] = append_keys[k]


def load_dataset_base(task, data_folder):
    """
    Returns a dictionary:
        {
            'train' : [{key:value}], # training dataset
            'test' : [{key:value}], # test dataset
        }

    Each of the train/test datasets are a list of samples. Each sample is a dictionary of following (key, value) pairs:
    {'bl': string, ques':string, 'txt':string, 'mdl':string, 'gr': int}


    The keys above are:
    bl: unique database like key to identify student response
    ques: The question to be answered
    'txt' : student response text to be scored
    'mdl': model or reference answer
    'gr' : median score of 2 human rater
    """

    dir_name = data_folder + "/" + task + '/'

    data = {}
    filenames = [("train", "train"), ("test", "test")]

    for i in range(len(filenames)):
        filename = os.path.join(dir_name + "{}.json".format(filenames[i][0]))
        with open(filename, "r") as f:
            da = json.load(f)
            data[filenames[i][1]] = da.get(filenames[i][0])

    compute_distribution(data)

    return data


def load_dataset_in_context_tuning(task, data_folder):
    data = load_dataset_base(task, data_folder)
    # Construct in_context examples from training dataset partitioned according to class score label
    examples_train = None
    if len(data['train_dist'].keys()) == 0:
        max_label = -1
        min_label = -1
    else:
        max_label = max(data['train_dist'].keys())
        min_label = min(data['train_dist'].keys())
    examples_train = {}
    for label in range(min_label, max_label + 1):
        examples_train[label] = []
    for datapoint in data["train"]:
        label = datapoint['gr']
        examples_train[label].append(datapoint)

    return data, examples_train, min_label, max_label


def load_meta_learning_dataset(data_folder):
    with open(data_folder + "/tasks.json", "r") as f:
        task_list = json.load(f)

    data_meta = {}
    for task in task_list:
        data_meta[task] = {}
        data, examples_train, min_label, max_label = load_dataset_in_context_tuning(task, data_folder)
        data_meta[task]["train"] = data["train"]
        data_meta[task]["test"] = data["test"]
        # Add in-context examples from training dataset => no information leakage from val/test sets
        data_meta[task]["examples"] = {}
        for label in range(min_label, max_label + 1):
            data_meta[task]["examples"][label] = examples_train[label]
        # Add task, min_label and max_label info to each sample
        for set in ["train", "test"]:
            for sample in data_meta[task][set]:
                sample["min"] = min_label
                sample["max"] = max_label
                sample["task"] = task

    # Union of training datasets across tasks
    data_meta["train"] = []
    for task in task_list:
        data_meta["train"] += data_meta[task]["train"]

    return data_meta


def tokenize_function(tokenizer, sentences_1, sentences_2=None):
    if (sentences_2 == None):
        return tokenizer(sentences_1, padding=True, truncation=True, return_tensors="pt")
    else:
        return tokenizer(sentences_1, sentences_2, padding=True, truncation=True, return_tensors="pt")


class CollateWraper:
    def __init__(self, tokenizer, data_meta, num_examples, trunc_len, mode, num_test_avg=1, test_batch_size=1,
                 max_seq_len=512):
        self.tokenizer = tokenizer
        self.data_meta = data_meta
        self.num_examples = num_examples
        self.trunc_len = trunc_len
        self.mode = mode
        # Adding an extra 50 words in case num_tokens < num_words after tokenization
        self.max_seq_len = max_seq_len + 50

        # Convert numeric scores to meaningful words
        self.label_to_text_arabic = {
            0: "ضعيف جدا",
            1: "ضعيف",
            2: "متوسط",
            3: "جيد",
            4: "جيد جدا",
            5: "ممتاز"
        }

        self.num_test_avg = num_test_avg
        self.test_batch_size = test_batch_size

    def __call__(self, batch):
        if (self.mode == "test"):
            # Since drop_last=False in test/val loader, record actual test_batch_size/val_batch_size for last batch constructed
            actual_batch_size = torch.tensor(len(batch)).long()
            # Repeat each test/val sample num_test_avg/num_val_avg times sequentially
            # augmentation (repeating num of test avg times)
            new_batch = []
            for d in batch:
                new_batch += [d for _ in range(self.num_test_avg)]
            batch = new_batch
        else:
            actual_batch_size = torch.tensor(-1).long()

        # Construct features: features_1 (answer txt) will have different segment embeddings than features_2 (remaining txt)
        features_1 = []
        features_2 = []
        for d in batch:
            # Randomly sample num_examples in-context examples from each class in train set for datapoint d
            examples_many_per_class = []
            # List examples_each_class stores one example from each class
            examples_one_per_class = []
            labels = list(range(d["min"], d["max"] + 1))

            for label in labels:
                # get examples for this particular score/grade
                examples_class = self.data_meta[d["task"]]["examples"][label]

                # Remove current datapoint d from examples_class by checking unique booklet identifiers => no
                # information leakage
                examples_class = [ex for ex in examples_class if ex["bl"] != d["bl"]]

                # Sampling num_examples without replacement
                # if there are not enough examples, put all off them after shuffling
                if (len(examples_class) < self.num_examples):
                    random.shuffle(examples_class)
                    examples_class_d = examples_class
                else:
                    # else sample (num_examples) from the examples
                    examples_class_d = random.sample(examples_class, self.num_examples)
                # check after sampling
                # if theres more than one example per class
                # add first one to one_per_class and the rest to many_per_class
                if (len(examples_class_d) > 1):
                    examples_one_per_class += [examples_class_d[0]]
                    examples_many_per_class += examples_class_d[1:]
                # if theres only one then many_per_class is empty
                elif (len(examples_class_d) == 1):
                    examples_one_per_class += [examples_class_d[0]]
                    examples_many_per_class += []
                else:
                    examples_one_per_class += []
                    examples_many_per_class += []

            # Construct input text with task instructions
            input_txt = "قيِّم هذه الإجابة: " + d['txt']
            features_1.append(input_txt)

            if label == -1:
                features_2 = None
            else:
                # Add range of valid score classes for datapoint d
                examples_txt = " تقييم: " + " ".join(
                    [(self.label_to_text_arabic[label] + " ") for label in range(d["min"], d["max"] + 1)])
                # Add question text
                examples_txt += "[SEP] السؤال: {} ".format(d["ques"])
                # add model answer
                examples_txt += "[SEP] النموذج: {} [SEP]".format(d["mdl"])

                # Shuffle examples across classes
                random.shuffle(examples_one_per_class)
                random.shuffle(examples_many_per_class)

                # Since truncation might occur if text length exceed max input length to LM,
                # we ensure at least one example from each score class is present
                examples_d = examples_one_per_class + examples_many_per_class
                curr_len = len(input_txt.split(" ") + examples_txt.split(" "))
                for i in range(len(examples_d)):
                    example = examples_d[i]
                    example_txt_tokens = example['txt'].split(" ")
                    curr_example_len = len(example_txt_tokens)
                    example_txt = " ".join(example_txt_tokens[:self.trunc_len])
                    example_label = (example['gr'])
                    # [SEP] at the end of the last example is automatically added by tokenizer
                    if (i == (len(examples_d) - 1)):
                        examples_txt += (
                                    " مثال: " + example_txt + " تقييم: " + self.label_to_text_arabic[example_label])
                    else:
                        examples_txt += (" مثال: " + example_txt + " تقييم: " + self.label_to_text_arabic[
                            example_label] + " [SEP] ")

                    # Stop adding in-context examples when max_seq_len is reached
                    if ((curr_example_len + curr_len) > self.max_seq_len):
                        break
                    else:
                        curr_len += curr_example_len
                features_2.append(examples_txt)
        self.features_1 = features_1
        self.features_2 = features_2

        inputs = tokenize_function(self.tokenizer, features_1, features_2)

        # Construct labels
        labels = torch.tensor([(d['gr']) for d in batch]).long()
        inputs['labels'] = labels

        # Store max_label for each d in batch which is used during softmax masking
        max_labels = torch.tensor([(d["max"] - d["min"] + 1) for d in batch]).long()
        return {
            "inputs": inputs,
            "max_labels": max_labels,
            "actual_batch_size": actual_batch_size
        }


