import os
import json

import pandas as pd


def get_json(df_train, df_test):
    # group by 'question id'
    questions_train = df_train.groupby('Question_ID')
    questions_test = df_test.groupby('Question_ID')

    # Define a common function to convert a DataFrame to a JSON string
    def df_to_json(df, split):
        data_entries = []
        for _, row in df.iterrows():
            entry = {
                'bl': str(row['Column1']),
                'ques': row['Question_Arabic'],
                'txt': row['Answer_Arabic'],
                'mdl': row['Model_Arabic'],
                'gr': int(row['Median']),
                'tsk': int(row['Question_ID']),
            }
            data_entries.append(entry)
        json_object = {split: data_entries}
        return json.dumps(json_object, ensure_ascii=False)

    json_train = [df_to_json(group, 'train') for _, group in questions_train]
    json_test = [df_to_json(group, 'test') for _, group in questions_test]

    return json_train, json_test


def build_data():
    csv_dir = 'data/csv/'
    data_dir = 'data/incml/'
    train_paths = [f'{csv_dir}q{q}/q{q}_trainset.csv' for q in range(1, 6)]
    test_paths = [f'{csv_dir}q{q}/q{q}_testset.csv' for q in range(1, 6)]

    qs_train = []
    qs_test = []

    # Read CSV files into DataFrames
    for train_path, test_path in zip(train_paths, test_paths):
        qs_train.append(pd.read_csv(train_path))
        qs_test.append(pd.read_csv(test_path))

    # get answers and scores for each question df
    # separate answers by score and save to file

    for q in range(0, 5):
        json_train, json_test = get_json(qs_train[q], qs_test[q])
        for i in range(0, len(json_train)):
            train_dict = json.loads(json_train[i])
            tsk_num = train_dict['train'][0]['tsk']
            task_dir = f'{data_dir}q{q + 1}/task{tsk_num}'
            os.makedirs(task_dir, exist_ok=True)
            file_path = f'{task_dir}/train.json'
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json_file.write(json_train[i])

        json_str = json.dumps({"train": []})  # Create an empty JSON string
        task_dir = f'{data_dir}q{q + 1}/task1'
        os.makedirs(task_dir, exist_ok=True)
        file_path = f'{task_dir}/train.json'

        with open(file_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_str)

        for i in range(0, len(json_test)):
            test_dict = json.loads(json_test[i])
            tsk_num = test_dict['test'][0]['tsk']
            task_dir = f'{data_dir}q{q + 1}/task{tsk_num}'
            os.makedirs(task_dir, exist_ok=True)
            file_path = f'{task_dir}/test.json'
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json_file.write(json_test[i])


if __name__ == '__main__':
    build_data()
