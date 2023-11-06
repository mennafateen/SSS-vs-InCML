# Step 2: Input: different (answer, score) tuples
# Output: (ans1, ans2, sim=sco1/sco2) tuples for similarity learning
import os

import numpy as np


def generate_pairs(ans1, ans2, sim, limit):
    if ans1.shape[0] == 0 or ans2.shape[0] == 0:
        return None
    pairs = []
    for i in range(ans1.shape[0]):
        for j in range(ans2.shape[0]):
            pairs.append((ans1[i], ans2[j], sim))
    if len(pairs) < limit:
        selected_pairs = [pairs[np.random.randint(len(pairs))] for _ in range(limit)]
        pairs = np.array(selected_pairs)
    elif len(pairs) >= limit:
        pairs = np.array(pairs[:limit])
    if len(pairs) == 0:
        return None
    return pairs


def pair_building(pt_0, pt_1, pt_2, pt_3, pt_4, pt_5, model_ans, limit, weights=False):
    all_pts_shape = [pt_0.shape[0], pt_1.shape[0], pt_2.shape[0], pt_3.shape[0], pt_4.shape[0], pt_5.shape[0]]
    total = np.sum(all_pts_shape)
    if len(pt_5) == 0:
        pt_5 = model_ans
    score_sims = [0.0, 0.2, 0.5, 0.6, 0.8, 1.0]
    datasets = []
    if weights:
        weights = [(shape / total) for shape in all_pts_shape]
    else:
        weights = [1, 1, 1, 1, 1, 1]
    for i, weight in enumerate(weights):
        dataset = generate_pairs([pt_0, pt_1, pt_2, pt_3, pt_4, pt_5][i], pt_5, score_sims[i], int(limit * weight))
        if dataset is not None:
            datasets.append(dataset)
    dataset = np.concatenate(datasets, axis=0)
    return dataset


if __name__ == '__main__':
    limits = [30, 50, 50]
    weights = [False, False, True]
    filenames = ['dataset_score_based_30', 'dataset_score_based_50', 'dataset_score_based_weights_50']
    for i, limit in enumerate(limits):
        print("Limit: ", limit)
        print("Weights: ", weights[i])
        print("Filename: ", filenames[i])
        print("---------------------------------------------------")
        for q_count in range(1, 6):
            path = f"data/sss/q{str(q_count)}/"
            subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
            print("Question: ", q_count)
            datasets = []
            for folder in subfolders:
                count = folder[-1:] if folder[-2] == 'k' else folder[-2:]
                if count.isnumeric() and int(count) != 1:
                    print("Subquestion: ", count)
                    pt_0, pt_1, pt_2, pt_3, pt_4, pt_5 = [np.load(f"{folder}/train/score{j}.npy") for j in range(6)]
                    print(pt_0.shape, pt_1.shape, pt_2.shape, pt_3.shape, pt_4.shape, pt_5.shape)
                    model_ans = np.load(f"{folder}/model_ans_train.npy")
                    ds = pair_building(pt_0, pt_1, pt_2, pt_3, pt_4, pt_5, model_ans, limits[i], weights[i])
                    datasets.append(ds)
            dataset = np.concatenate(datasets, axis=0)
            np.save(path + f'/{filenames[i]}', dataset)
            print("---------------------------------------------------")
