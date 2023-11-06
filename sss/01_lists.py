# Step 1: Separate answers into score-based lists (0-5 points)
# to create (answer1, answer2, similarity=score1/score2) tuples for training.

import os

import numpy as np
import pandas as pd


def extract_answers_scores(df):
    # first concatenate question and answer columns together
    df['Question_Answer'] = df['Question_Arabic'].str.cat(df['Answer_Arabic'], sep="[SEP]")
    qa = np.array(df['Question_Answer'].values)
    scores = np.array(df['Median'].values)

    return qa, scores


def separate_by_score(all_answers_with_questions, all_scores):
    answer_score = []
    for answers, scores in zip(all_answers_with_questions, all_scores):
        answer_score.append((answers, scores))
    df_np = np.array(answer_score)

    pt = [[] for _ in range(6)]

    for ans_sco in df_np:
        score = int(ans_sco[1])
        if 0 <= score <= 5:
            pt[score].append(ans_sco[0])

    return pt[0], pt[1], pt[2], pt[3], pt[4], pt[5]


def process_data(data_type, question_sets=5):
    def read_question_set_csv(prefix, num_sets):
        return [pd.read_csv(f'data/csv/q{q}/q{q}_{data_type}set.csv') for q in range(1, num_sets + 1)]

    def separate_and_save_scores(path, answers, scores):
        score_arrays = [np.array(pt) for pt in separate_by_score(answers, scores)]
        for i, pt in enumerate(score_arrays):
            np.save(os.path.join(path, f'score{i}.npy'), pt)

    for q_count in range(1, question_sets + 1):
        print(f'Data Type: {data_type}, Question: {q_count}')

        # Load the dataset
        dataset = pd.read_csv(f'data/csv/q{q_count}/q{q_count}_{data_type}set.csv')

        # Group by Question_ID
        q_dict = dataset.groupby('Question_ID')

        # Get list of question IDs
        q_ids = list(q_dict.groups.keys())

        qs = [q_dict.get_group(q_id) for q_id in q_ids]

        for df in qs:
            question_id = str(df['Question_ID'].iloc[0])

            path = f"data/sss/q{q_count}/task{question_id}/{data_type}/"
            os.makedirs(path, exist_ok=True)

            answers, scores = extract_answers_scores(df)
            separate_and_save_scores(path, answers, scores)


def process_model_answers(data_type, question_sets=5):
    for q_count in range(1, question_sets + 1):
        print(f'Data Type: {data_type}, Question: {q_count}')

        # Load the dataset
        dataset = pd.read_csv(f'data/csv/q{q_count}/q{q_count}_{data_type}set.csv')

        # Group by Question_ID
        q_dict = dataset.groupby('Question_ID')

        # Get list of question IDs
        q_ids = list(q_dict.groups.keys())

        qs = [q_dict.get_group(q_id) for q_id in q_ids]

        for df in qs:
            question_id = str(df['Question_ID'].iloc[0])
            ans = df['Question_Arabic'].tolist()[0] + ' [SEP] ' + df['Model_Arabic'].tolist()[0]
            model_ans = np.array([ans])
            question_id = str(df['Question_ID'].iloc[0])
            path = f"data/sss/q{str(q_count)}/task{question_id}/model_ans_{data_type}.npy"
            np.save(path, model_ans)


if __name__ == '__main__':
    process_data("train")
    process_data("test")

    process_model_answers("train")
    process_model_answers("test")




