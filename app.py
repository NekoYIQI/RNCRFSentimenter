import os
import operator
import urllib.request
import bs4 as bs
from flask import Flask, render_template, request

app = Flask(__name__)

from sentimenter import RNCRFSentimenter
from nltk import CoreNLPDependencyParser
from util.dtree_util import *
from string import whitespace
import pickle
import numpy as np
from joblib import dump, load

# load crf model
crf = load('util/crf.joblib') 

# data preparation
params = 'util/data/final_params_sample_2'
[Wr_dict, Wv, b, we], vocab, rel_list = pickle.load(open(params, 'rb'))

# get word2vec dictionary
word2vec_file = open('util/data/word2vec_mc5.txt', 'r')
word2vec = word2vec_file.readlines()
word2vec_file.close()
dic = {}
    
index = 0
for line in word2vec:
    if index == 0:
        index += 1
        continue
    index += 1
    word_vector = line.split(" ")
    word = word_vector[0]

    vector_list = []
    for element in word_vector[1:]:
        vector_list.append(float(element))

    vector = np.asarray(vector_list)
    dic[word] = vector

# connect to coreNLP server    
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

def generate_result(sents, predictions):
    COLOR = ['orange', 'violet']
    sentences = []
    # print(sents)
    # print(predictions)
    for i in range(len(sents)):
        print(predictions[i])
        sentence = ""
        for j in range(len(sents[i])):
            if predictions[i][j] == '0':
                sentence += sents[i][j]
                sentence += " "
            elif predictions[i][j] == '1' or predictions[i][j] == '2':
                sentence += "<strong><span style='color:%s'>" % COLOR[0]
                sentence += sents[i][j]
                sentence += "</span></strong>"
                sentence += " "
            elif predictions[i][j] == '3' or predictions[i][j] == '4':
                sentence += "<strong><span style='color:%s'>" % COLOR[1]
                sentence += sents[i][j]
                sentence += "</span></strong> "
        sentences.append(sentence)
    print(sentences)
    return sentences

@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        # get url that the person has entered
        try:
            url = request.form['url']
            sauce = urllib.request.urlopen(url).read()
        except:
            errors.append(
                "Unable to get URL. Please make sure it's valid and try again."
            )
            return render_template('index.html', errors=errors)
        
        if url:
            soup = bs.BeautifulSoup(sauce, 'lxml')
            sentences = []
            text = []
            for paragraph in soup.find_all('p'):
                if paragraph.text[0] not in whitespace:
                    text.append(paragraph.text.strip())
                    # print(paragraph.text.strip())
            for item in text:
                item = item.strip()
                l = item.split('. ')
                for sent in l:
                    sentences.append(sent.replace("\n", '').replace('\r', ''))
            print(sentences)
            # text processing
            rncrf = RNCRFSentimenter(dep_parser, Wr_dict, Wv, b, we, vocab, dic)
            sents, predictions = rncrf.analyse(crf, sentences)
            results = generate_result(sents, predictions)
            
    return render_template('index.html', errors=errors, results=results)


if __name__ == '__main__':
    app.run()