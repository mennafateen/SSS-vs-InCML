# sss-vs-incml

## Introduction

This repository contains the code used for the experiments in the paper *"In-Context Meta-Learning vs. Semantic Score-Based Similarity: A Comparative Study in Arabic Short Answer Grading"* by Fateen et. al. (2023).

## Requirements

To install requirements:

1. Create a conda environemnt with specified python version
```
conda create -n sss-vs-incml python=3.9
```
2.Install the required libraries by running
```
pip install -r requirements.txt
```

## Code Organization

The code is organized as follows:
1. `data/`: contains the data used in the experiments.
2. `sss/`: contains the code for the SSS model. The files can be run in the order defined by the file names.
3. `incml/`: contains the code for the INCML model. The files can be run in the order defined by the file names.
4. `prompt_classify.py`: the code for the prompt classification model.

## Data
The data used in the experiments was collected from the [Arabic ASAG dataset](https://aclanthology.org/2020.lrec-1.321.pdf)
We divided the data per prompt type and divided them into train/test sets. The data can be found in the `data/csv` folder.

## Citation
If you use this code in your research, please cite the following paper:
```
@inproceedings{fateen2023incontext,
  title={In-Context Meta-Learning vs. Semantic Score-Based Similarity: A Comparative Study in Arabic Short Answer Grading},
  author={Fateen, Menna and Mine, Tsunenori},
  year={2023}
}
```
