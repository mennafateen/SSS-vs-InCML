import os
import math

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import cohen_kappa_score, accuracy_score

from utils import pta

os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


for q_count in range(1, 6):

    print('Question:' + str(q_count))
    directory = 'data/sss/q' + str(q_count)

    # Load three different datasets for each question
    dataset_1 = np.load(directory + '/dataset_score_based_30.npy')
    dataset_2 = np.load(directory + '/dataset_score_based_50.npy')
    dataset_3 = np.load(directory + '/dataset_score_based_weights_50.npy')

    y_true = []
    y_pred = []

    test_samples = []
    output_path_1 = 'models/sss/q' + str(q_count) + '/' + 'model_30'
    output_path_2 = 'models/sss/q' + str(q_count) + '/' + 'model_50'
    output_path_3 = 'models/sss/q' + str(q_count) + '/' + 'model_50_weights'

    print("Model 0: loading model from:", output_path_1)
    print("Model 1: loading model from:", output_path_2)
    print("Model 2: loading model from:", output_path_3)

    unseen_ans_qwk = 0
    unseen_ans_pta = [0, 0, 0, 0]
    unseen_ques_qwk = 0
    unseen_ques_pta = [0, 0, 0, 0]

    ans_true = []
    ans_pred = []

    model_1 = SentenceTransformer(output_path_1)
    model_2 = SentenceTransformer(output_path_2)
    model_3 = SentenceTransformer(output_path_3)

    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    models = [model_1, model_2, model_3]

    for num, model in enumerate(models):
        for folder in subfolders:
            if folder[-2] == 'k':
                count = folder[-1:]
            else:
                count = folder[-2:]
            if count.isnumeric():
                if int(count) != 1:  # task 1 is not included in training dataset (unseen question)
                    dataset = np.load(folder + '/dataset_score_based.npy')
                    y_true = []
                    y_pred = []
                    for row in dataset:
                        emb1 = model.encode([row[0]])
                        #                     print(row[0])
                        emb2 = model.encode([row[1]])
                        #                     print(row[1])
                        cos_sim = util.cos_sim(emb1, emb2)
                        y_true.append(float(row[2]) * 5)
                        ans_true.append(float(row[2]) * 5)
                        y_pred.append(math.floor(cos_sim * 5))
                        ans_pred.append(math.floor(cos_sim * 5))
                    unseen_ans_qwk += cohen_kappa_score(y_true, y_pred, weights='quadratic')
                    test_ans_pta = pta(y_true, y_pred)
                    for i in range(4):
                        unseen_ans_pta[i] += test_ans_pta[i]

                elif int(count) == 1:
                    dataset = np.load(folder + '/dataset_score_based.npy')
                    y_true = []
                    y_pred = []
                    for row in dataset:
                        emb1 = model.encode([row[0]])
                        emb2 = model.encode([row[1]])
                        cos_sim = util.cos_sim(emb1, emb2)
                        y_true.append(float(row[2]) * 5)
                        y_pred.append(math.floor(cos_sim * 5))
                    unseen_ques_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
                    unseen_ques_pta = pta(y_true, y_pred)
                    print(f"Model {num} - Unseen question QWK:  {unseen_ques_qwk}")
                    print(f"Model {num} - Unseen question PTA: {unseen_ques_pta}")

        print(f"Model {num} - Unseen answer QWK: {cohen_kappa_score(ans_true, ans_pred, weights='quadratic')}")
        print(f"Model {num} - Unseen answer PTA: {pta(ans_true, ans_pred)}")


