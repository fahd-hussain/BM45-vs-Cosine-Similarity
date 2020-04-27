import nltk
from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

class QUERY_PARSE:
    def __init__(self):
        self.stopwords = open('Stopword-List.txt', encoding='utf8', errors='ignore').read().split()
        self.removetable = str.maketrans("", "", "'!@#$%^&*()_=-\|][:';:,<.>/?`~")
        self.ps = PorterStemmer()

    def parse(self, query):
        tokens=nltk.word_tokenize(query)

        tokens = [x.translate(self.removetable) for x in tokens]
        tokens = [element.lower() for element in tokens]
        tokens = [x for x in tokens if x.isalnum() and x not in self.stopwords]
        # tokens = [self.ps.stem(element) for element in tokens]
        return tokens
