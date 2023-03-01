from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline, flatten
from nltk.lm import MLE
from nltk.corpus import stopwords
import numpy as np
import argparse
import random
import string
#import nltk
#nltk.download('stopwords')

parser = argparse.ArgumentParser(description='Author Text Classification')
parser.add_argument('arg1', help='training data')
parser.add_argument('-test', help='Test file')
args = parser.parse_args()
author_file = args.arg1
test_file = args.test

def score_dev_set(lm, text, authors_dict, author_name, i):    
    predictions = np.array([1 if lm.score(x) > 0.5 else 0 for x in text])
    true_labels = np.array([1 if x in authors_dict[i] else 0 for x in text])
    correct_preds = np.sum(predictions==true_labels)
    accuracy = correct_preds*100/len(text)
    #print(f'correct preds:{correct_preds} len text:{len(text)} predictions sum:{np.sum(predictions)}')
    print(f'{author_name}     {accuracy}% correct')
    return 0

if __name__=='__main__':
    #Preprocessing
    file_list = ['./data/austen_utf8.txt', './data/dickens_utf8.txt', './data/tolstoy_utf8.txt', './data/wilde_utf8.txt']
    train_texts = {}
    dev_authors = {}
    dev_texts = []
    authors_list = ['austen', 'dickens', 'tolstoy', 'wilde']

    for i, file in enumerate(file_list):
        f = open(file, 'r')
        train_texts[i] = []
        dev_authors[i] = []
        for x in f:
            r = random.random()
            if r<0.9:
                x = x.replace(string.punctuation, '')
                added_line = [w for w in x if w.lower() not in stopwords.words('English')]
                train_texts[i] += added_line #you were adding strings not lists. need to add sentence start & end symbols to b/w each line
            else:
                x = x.replace(string.punctuation, '')
                added_line = [w for w in x if w.lower() not in stopwords.words('English')]
                dev_authors[i] += added_line
                dev_texts += added_line
        f.close()

    #Train Models
    lm_models = {}
    for i in range(len(train_texts)):
        train, vocab = padded_everygram_pipeline(2, train_texts[i])
        lm = MLE(2)
        lm.fit(train, vocab)
        lm_models[i] = lm
        score_dev_set(lm, dev_texts, dev_authors, authors_list[i], i)

    #Testing
    file_names = ['austens', 'dickens', 'tolstoy', 'wilde'] #comment out in final implementation & create author file
    test_sent = open(test_file, 'r')
    for sentence in test_sent:
        best_score = np.Inf
        prediction = ''
        for i in range(len(lm_models)):
            model = lm_models[i]
            perplx_score = model.perplexity(sentence)  #to do:calculate according to the lm_model[i]
            if perplx_score < best_score:
                best_score = perplx_score
                prediction = file_names[i] 
        print(prediction)
