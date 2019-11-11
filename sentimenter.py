from util.dtree_util import *
from nltk import CoreNLPDependencyParser
from collections import OrderedDict
import util.crf_propagation as prop
import numpy as np
import pycrfsuite

class RNCRFSentimenter():

	def __init__(self, dep_parser, Wr_dict, Wv, b, we, vocab, dic):
		self.dep_parser = dep_parser
		self.Wr_dict = Wr_dict
		self.Wv = Wv
		self.b = b
		self.we = we
		self.vocab = vocab
		self.dic = dic
		self.dim = 100
		self.num_class = 5

	def rawparse2deptree(self, parse):
	    conll_list = parse.to_conll(4).split('\n')
	    # remove the last empty entry from the conll list
	    conll_list = conll_list[:-1]
	    dep_list = []
	    for item in conll_list:
	        dep = item.split('\t')
	        # print(dep)
	        if len(dep) > 0:
	            dep_list.append(dep)
	    # print(dep_list)

	    max_index = len(dep_list)
	    # print("length of dep list: ", max_index)
	    # for i in range(0, max_index + 1):
	    #     print(i)
	    nodes = ['ROOT']
	    for i in range(max_index):
	        nodes.append(dep_list[i][0])
	    tree = dtree(nodes)
	    
	    for i in range(1, max_index + 1):
	        dep = dep_list[i - 1]
	        par_index = int(dep[2])
	        kid_index = i
	        rel = dep[3]
	        tree.add_edge(par_index, kid_index, rel)
	    # tree.get_tree()

	    for node in tree.get_nodes():
	    	if node.word.lower() in self.vocab:
	    		node.ind = self.vocab.index(node.word.lower())

	    return tree

	def sentences2deptree(self, sentences):
		trees = []
		for sentence in sentences:
			parse, = self.dep_parser.raw_parse(sentence)
			trees.append(self.rawparse2deptree(parse))
		return trees

	def get_hidden_inputs(self, tree):
		nodes = tree.get_nodes()

		for node in nodes:
			node.vec = self.we[:, node.ind].reshape((self.dim, 1))

		param_list = (self.Wr_dict, self.Wv, self.b, self.we)
		prop.forward_prop(param_list, tree, self.dim, self.num_class)
		
		sent = []
		h_input = np.ones((len(tree.nodes) - 1, self.dim))

		for ind, node in enumerate(tree.nodes):
			if ind != 0:
				if tree.get(ind).is_word == 0:
					sent.append(None)
					for i in range(self.dim):
						h_input[ind - 1][i] = 0
				else:
					sent.append(node.word)
					for i in range(self.dim):
						h_input[ind - 1][i] = node.p[i]

		return sent, h_input

	def word2features(self, sent, h_input, i):
    
	    features = OrderedDict() 
	    
	    features['bias'] = 1
	    
	    if sent[i] == None:
	        features['punkt'] = 1
	    else:
	        for n in range(self.dim):
	            features['we=%d' % n] = h_input[i][n]
	    # add features for the left word of current word
	    if i > 0 and sent[i - 1] == None:
	        features['-1punkt'] = 1
	    elif i > 0:
	        for n in range(self.dim):
	            features['-1we=%d' % n] = h_input[i - 1][n]
	    else:
	        # if there is no left word, mark current word as the beginning of sentence
	        features['BOS'] = 1
	    
	    # add features for the right word of current word
	    if i < (len(sent) - 1) and sent[i + 1] == None:
	        features['+1punkt'] = 1
	    elif i < (len(sent) - 1):
	        for n in range(self.dim):
	            features['+1we=%d' % n] = h_input[i + 1][n]
	    else:
	        features['EOS'] = 1
	                
	    return features


	def sent2features(self, sent, h_input):
	    return [self.word2features(sent, h_input, i) for i in range(len(sent))]


	def analyse(self, crf, sentences):
	    # output labels
	    
	    trees = self.sentences2deptree(sentences)

	    sents = []
	    
	    X = []
	    for ind, tree in enumerate(trees):
	        nodes = tree.get_nodes()
	        sent = []
	        h_input = np.zeros((len(tree.nodes) - 1, self.dim))
	        y_label = np.zeros((len(tree.nodes) - 1,), dtype = int)
	        
	        for index, node in enumerate(nodes):
	            if node.word.lower() in self.vocab:
	                node.vec = self.we[:, node.ind].reshape((self.dim, 1))
	            elif node.word.lower() in self.dic.keys():
	                node.vec = self.dic[node.word.lower()].reshape(self.dim, 1)
	            else:
	                node.vec = np.random.rand(self.dim, 1)
	            
	        prop.forward_prop([self.Wr_dict, self.Wv, self.b, self.we], tree, self.dim, self.num_class, labels=False)
	        
	        sent, h_input = self.get_hidden_inputs(tree)
	        sents.append(sent)

	        crf_sent_features = self.sent2features(sent, h_input)
	        
	        # prepare input data for crf
	        X.append(crf_sent_features)
	    preds = crf.predict(X)        

	    return sents, preds
        

