from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline, flatten
import numpy as np
from nltk.lm import MLE
import argparse

parser = argparse.ArgumentParser(description='Author Text Classification')

parser.add_argument('arg1', help='training data')
parser.add_argument('-test', help='Test file')

args = parser.parse_args()

#Preprocessing
file_list = ['./data/austen_utf8.txt', './data/dickens_utf8.txt', './data/tolstoy_utf8.txt', './data/wilde_utf8.txt']
data_path = '/Users/ritikparkar/UMN/Courses/CSCI 5541/HW2/NGram-Language-Models/Data/'
text_file = {}
#to do: only utf files
for i, file in enumerate(file_list):
    f = open(file, 'r')
    text_file[i] = ''
    for x in f:
        # print(x)
        text_file[i] += f.readline()

lm_models = {}
for i in range(len(text_file)):
    train, vocab = padded_everygram_pipeline(2, text_file[i])
    
    #Training
    lm = MLE(2)
    lm.fit(train, vocab)
    print(lm.vocab)
    lm_models[i] = lm

#Edit the above code to remove spaces, punctuations, stop words. 