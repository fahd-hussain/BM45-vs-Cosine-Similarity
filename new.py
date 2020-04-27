from corpusParse import *
from queryParse import *
from cosine import *
from okapi import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import math

def main():
    docParse = CORPUS_PARSE(filename='200_content.txt')
    queryParse = QUERY_PARSE()
    tokens, docTokens, documents = docParse.parse()

    okapi_sim = OKAPI(corpus=docTokens)

    # Dictionary.txt
    f = open("dictionary.txt", "a", encoding='UTF8')
    for i in range(len(tokens) - 1):
        if (tokens[i] != tokens[i+1]):
            f.write(tokens[i] + "\n")
    f.close()

    queryCount = 1
    while True:
        query = input('Enter your query: \n')
        f = open(f"okaquery{queryCount}.result.txt", "a", encoding='UTF8')
        f.write(query + "\n")
        results = okapi_sim.run(query)
        for result in results:
            sorted_x = sorted(result.items(), key=operator.itemgetter(1))
            sorted_x.reverse()
            for i in sorted_x[:10]:
                print(i[0], i[1])
                f.write(str(i[0]) + "\t" + str(i[1]) + "\n")
            f.close()
        yesno = input('Do you want to continue? press y for yes, n for no\n')
        if yesno == 'n':
            break
        queryCount += 1

main()
