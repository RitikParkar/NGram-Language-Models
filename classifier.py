from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline, flatten
import numpy as np
from nltk.lm import MLE
import argparse

parser = argparse.ArgumentParser(description='Author Text Classification')

parser.add_argument('arg1', help='training data')
parser.add_argument('-test', help='Test file')

args = parser.parse_args()

author_file = args.arg1
test_file = args.arg2

file_names = []
auth_f = open(author_file, 'r')
for x in auth_f:
    file_names.append(x)

#Preprocessing
file_list = ['./data/austen_utf8.txt', './data/dickens_utf8.txt', './data/tolstoy_utf8.txt', './data/wilde_utf8.txt']
data_path = '/Users/ritikparkar/UMN/Courses/CSCI 5541/HW2/NGram-Language-Models/Data/'
final_texts = {}
#to do: only utf files
for i, file in enumerate(file_list):
    f = open(file, 'r')
    final_texts[i] = ''
    for x in f:
        # print(x)
        final_texts[i] += f.readline()

lm_models = {}
for i in range(len(final_texts)):
    train, vocab = padded_everygram_pipeline(2, final_texts[i])
    
    #Training
    lm = MLE(2)
    lm.fit(train, vocab)
    print(lm.vocab)
    lm_models[i] = lm

#Edit the above code to remove spaces, punctuations, stop words. 

#Testing
file_names = ['austens', 'dickens', 'tolstoy', 'wilde'] #comment out in final implementation 
test_sent = open(test_file, 'r')
for sentence in test_sent:
    best_score = np.Inf
    prediction = ''
    for i in range(len(lm_models)):
        model = lm_models[i]
        perplx_score = 0  #to do:calculate according to the lm_model[i]
        if perplx_score < best_score:
            best_score = perplx_score
            prediction = file_names[i] 
    print(prediction)
