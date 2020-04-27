import nltk
from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

class CORPUS_PARSE:
	def __init__(self, filename):
		self.filename = filename
		self.stopwords = open('Stopword-List.txt', encoding='utf8', errors='ignore').read().split()
		self.removetable = str.maketrans("", "", "'!@#$%^&*()_=-\|][:';:,<.>/?`~")
		self.tokens = {}
		self.docToken = {}
		self.ps = PorterStemmer()

	def parse(self):
		documents = open(self.filename, encoding='utf8', errors='ignore').read()
		# Tokenize whole documents
		tokens=nltk.word_tokenize(documents)

		tokens = [x.translate(self.removetable) for x in tokens]
		tokens = [element.lower() for element in tokens]
		# tokens = [self.ps.stem(element) for element in tokens]
		tokens = [x for x in tokens if x.isalnum() and x not in self.stopwords]
		# tokens = [element.lower() for element in tokens]
		tokens = sorted(tokens)
		self.tokens = tokens
		# print(tokens)

		singleDoc = documents.split('\n')
		docToken = {}
		for x in range(len(singleDoc)):
			docToken[x] = nltk.word_tokenize(singleDoc[x])

		# Remove Special characters Doc Wise
		for x in range(len(singleDoc)):
			docToken[x] = [y.translate(self.removetable) for y in docToken[x]]
		# doc wise sorted
		for x in range(len(singleDoc)):
			docToken[x] = sorted(docToken[x])
		# Decaptilized Doc Wise
		for x in range(len(singleDoc)):
			docToken[x] = [element.lower() for element in docToken[x]]
			# docToken[x] = [self.ps.stem(element) for element in tokens]
		for x in range(len(singleDoc)):
			docToken[x] = [y for y in docToken[x] if y.isalnum() and y not in self.stopwords]

		self.docToken = docToken

		return tokens, docToken, documents
