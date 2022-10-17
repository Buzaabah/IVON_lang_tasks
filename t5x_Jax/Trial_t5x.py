import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.spatial
import matplotlib.pyplot as plt


# from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch
from pprint import pprint

from simplet5 import SimpleT5

import sys
import os
import glob

# We will use T5 trained using teacher forcing. This means that for training we always need an input sequence and
# a target sequence

PATH = 'input/mohler/mohler_dataset_edited.csv'

dataset = pd.read_csv(PATH)
print("Top five rows of data:\n", dataset.head())

print("Shape of data:\n", dataset.shape)

df = dataset.drop(['score_me', 'score_other', 'score_avg'], axis=1)
print("Top five rows of data:\n", df.head())


"""
- T5 extract the answer from the question, so here we should feed the model with the question and the exact answer/s.
- We add the student answers to desired answers, and rename the dataset columns to source_text and target_text,
which is adapted for modeling.
"""
# list to feed to the model
# 1. source_text
q_list = "question: " + df['question']

# 2. target_text
ans_list = df['desired_answer'] + df['student_answer']

dict_data = {'source_text': q_list, 'target_text': ans_list}

# dictionary to dataflame
df = pd.DataFrame(dict_data)

print("The seq to seq  data for modeling:\n", df.head())

print("Question in position 1:\n", df['source_text'][0])
print("Answer in position 1:\n", df['target_text'][0])

# shape of dataset, unique (non duplicate) length of source and target text
print("Shape of seq to seq data for modelling :", df.shape)
print("length of unique source text (questions):", len(df.source_text.unique()))
print("length of unique target text (answers):", len(df.target_text.unique()))

# Split into train and test data

train_data, val_data = train_test_split(df[:-100], test_size=0.2)
test_data = df[-100:]

print("Shape of training seq to seq data :", train_data.shape)
print("Shape of validation seq to seq data :", val_data.shape)
print("Shape of test seq to seq data :", test_data.shape)


model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
model.train(train_df = train_data,
            eval_df = val_data,
            source_max_token_len=128,
            target_max_token_len=50,
            batch_size=8, max_epochs=3, use_gpu=False)