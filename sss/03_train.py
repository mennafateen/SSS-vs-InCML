# -*- coding: utf-8 -*-
# step 2_1 train the BERT in a similarity training way using SBERT library
# input: (ans1, ans2, sim) tuples generated from step 1_2
# output: a trained BERT model

import os
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import models
from torch.utils.data import DataLoader


def train_bert_model(q_count, dataset, output_path):
    # Parameter settings
    BATCH_SIZE = 16
    NUM_EPOCHS = 6
    WARMUP_STEPS = int(dataset.shape[0] // BATCH_SIZE * 0.1)

    # Load BERT model
    MODEL_NAME = 'aubmindlab/bert-base-arabertv02'
    transformer = models.Transformer(MODEL_NAME)
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode_mean_tokens=True,
                             pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[transformer, pooling])

    # Build the training dataloader
    train_data = [InputExample(texts=[s[0], s[1]], label=s[2].astype(np.float32)) for s in dataset]
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=NUM_EPOCHS,
              warmup_steps=WARMUP_STEPS,
              output_path=output_path
              )

    print(f'Training for Question {q_count} complete.')


if __name__ == '__main__':
    os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    for q_count in range(1, 6):
        print('Question:' + str(q_count))
        directory = 'data/sss/q' + str(q_count)

        # Load three different datasets for each question
        dataset_1 = np.load(directory + '/dataset_score_based_30.npy')
        dataset_2 = np.load(directory + '/dataset_score_based_50.npy')
        dataset_3 = np.load(directory + '/dataset_score_based_weights_50.npy')

        # Define three different output paths
        output_path_1 = 'models/sss/q' + str(q_count) + '/' + 'model_30'
        output_path_2 = 'models/sss/q' + str(q_count) + '/' + 'model_50'
        output_path_3 = 'models/sss/q' + str(q_count) + '/' + 'model_50_weights'

        if not os.path.exists(output_path_1):
            os.makedirs(output_path_1)

        if not os.path.exists(output_path_2):
            os.makedirs(output_path_2)

        if not os.path.exists(output_path_3):
            os.makedirs(output_path_3)

        train_bert_model(q_count, dataset_1, output_path_1)
        train_bert_model(q_count, dataset_2, output_path_2)
        train_bert_model(q_count, dataset_3, output_path_3)



