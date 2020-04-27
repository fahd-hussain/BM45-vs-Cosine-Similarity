from math import log
from invdx import *
import math

k1 = 1.2
k2 = k1
b = 0.75
R = 0.0

# n = occurance of query in documents
# N = number of document
# k = occurance of query in doc
# dl = length of document
# qf = query with 2 terms
# r and R = relevence information

class OKAPI:
	def __init__(self, corpus):
		self.index, self.dlt = build_data_structures(corpus)
		self.postingList()

	def postingList(self):
		f = open("postingList.txt", "w", encoding='UTF8')
		index = self.index.index
		for item in index:
			f.write(item + "\n")
			for docID in index[item]:
				tfw = math.log(index[item][docID], 10) + 1
				f.write(docID + "\t" + str(index[item][docID]) + "\t" + str(tfw) + "\n")
		f.close()

	def run(self, query):
		results = []
		results.append(self.run_query_okapi(query))
		return results

	def run_query_okapi(self, query):
		query_result = dict()
		index = self.index
		for term in query:
			if term in index:
				# retrieve index entry
				doc_dict = index[term]
				# for each document and its word frequency
				for docid, freq in doc_dict.items():
					# calculate score
					score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt), dl=self.dlt.get_length(docid), avdl=self.dlt.get_average_length())
					# this document has already been scored once
					if docid in query_result:
						query_result[docid] += score
					else:
						query_result[docid] = score
		return query_result

	def postingList(self):
		f = open("postingList.txt", "w", encoding='UTF8')
		index = self.index.index
		for item in index:
			f.write(item + "\n")
			for docID in index[item]:
				tfw = math.log(index[item][docID], 10) + 1
				f.write(docID + "\t" + str(index[item][docID]) + "\t" + str(tfw) + "\n")
		f.close()

def score_BM25(n, f, qf, r, N, dl, avdl):
		K = compute_K(dl, avdl)
		first = log(((r + 0.5) / (R - r + 0.5)) / ((n - r + 0.5) / (N - n - R + r + 0.5)))
		second = ((k1 + 1) * f) / (K + f)
		third = ((k2+1) * qf) / (k2 + qf)
		return first * second * third

def compute_K(dl, avdl):
	return k1 * ((1-b) + b * (float(dl)/float(avdl)))