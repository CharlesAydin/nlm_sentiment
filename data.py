import os
import re
import numpy as np

class PreProcessor():
	
	def __init__(self, path_to_aclimdb):
		self.path_to_aclimdb = path_to_aclimdb
		self.pos_files = os.listdir(self.path_to_aclimdb+"/train/pos")
		self.neg_files = os.listdir(self.path_to_aclimdb+"/train/neg")
		self.pos_text = []
		self.neg_text = []
		self.word_freq = {}
		self.word_ind = {}
		self.doc_id = np.array([])
		self.train_in = np.array([])
		self.train_out = np.array([])


	def read(self):
		for file in self.pos_files:
			if not '.DS' in file:
				with open(self.path_to_aclimdb+"/train/pos/"+file, 'r') as f:
					self.pos_text.append(re.findall("[a-zA-Z1234567890-]+", f.read().lower()))

		for file in self.neg_files:
			if not '.DS' in file:
				with open(self.path_to_aclimdb+"/train/neg/"+file, 'r') as f:
					self.neg_text.append(re.findall("[a-zA-Z1234567890-]+", f.read().lower()))

	def get_word_freq(self):
		for doc in self.pos_text:
			for word in doc:
				if not self.word_freq.get(word) is None:
					self.word_freq[word] += 1
				else:
					self.word_freq[word] = 1

		for doc in self.pos_text:
			for word in doc:
				if not self.word_freq.get(word) is None:
					self.word_freq[word] += 1
				else:
					self.word_freq[word] = 1

	def make_word_ind(self, min_freq = 25):
		self.word_ind['<unk>'] = 0
		self.word_ind['<card>'] = 1
		i = 2
		for word in self.word_freq.keys():
			if self.word_freq[word] >= min_freq:
				self.word_ind[word] = i
				i += 1
		self.convert_word_to_ind(self.pos_text)
		self.convert_word_to_ind(self.neg_text)

	def convert_word_to_ind(self, docs):
		for doc in range(len(docs)):
			for word in range(len(docs[doc])):
				if self.num_there(docs[doc][word]):
					docs[doc][word] = self.word_ind['<card>']
				else:
					if not self.word_ind.get(docs[doc][word]) is None:
						docs[doc][word] = self.word_ind[docs[doc][word]]
					else:
						docs[doc][word] = self.word_ind['<unk>']


	def find_ngrams(self, input_list, n):
  		return zip(*[input_list[i:] for i in range(n)])

	def num_there(self, s):
	    return any(i.isdigit() for i in s)

	def make_grams(self, gram_length = 17):
		din = []
		tin = []
		tout = []
		for doc in self.pos_text:
			for gram in self.find_ngrams(doc, gram_length):
				din.append(0)
				tin.append(gram[:-1])
				tout.append(gram[-1:])
		for doc in self.neg_text:
			for gram in self.find_ngrams(doc, gram_length):
				din.append(1)
				tin.append(gram[:-1])
				tout.append(gram[-1:])
		self.train_in = np.array(tin, dtype=int)
		self.train_out = np.array(tout, dtype=int)
		self.doc_id = np.array(din, dtype=int)


if __name__ == "__main__":
	p = PreProcessor('../aclImdb')
	p.read()
	p.get_word_freq()
	p.make_word_ind()
	p.make_grams()
	print p.train_in[:10]