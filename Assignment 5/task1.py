#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 19:29:22 2024

@author: aditib
"""

import re 
import numpy as np

#task1

trainfile = sc.textFile('s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt')
lines = trainfile.filter(lambda x : 'id' in x)

textkey = lines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:-7]))
pat = re.compile([^a-zA-Z]')
doc_word_dic = textkey.map(lambda x : (str(x[0]), pat.sub(' ', x[1]).lower().split())) #doc word dictionary
allwords = doc_word_dic.flatMap(lambda x: ((j, 1) for j in x[1]))
wordcounts = allwords.reduceByKey (lambda a, b: a + b).sortBy(lambda x: x[0])
top_words = wordcounts.top (20000, lambda x : x[1])

top_words_rank = sc.parallelize(range(20000))

freq_dict = top_words_rank.map (lambda rank: (top_words[rank][0], rank))
words_to_check = ["applicant", "and", "attack", "protein", "car"]

for word in words_to_check:
    print(word + " frequency is" + str(freq_dict.lookup(word)[0]))
    
#task 2

docIDAndWords = doc_word_dic.flatMap(lambda doc: [(word, doc[0]) for word in doc[1]])
wordCounts = docIDAndWords.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)
wordCountsBroadcast = sc.broadcast(dict(wordCounts.collect()))

def tfidfVectorize(doc):
    docID = doc[0]
    wordCountList = doc[1]
    nWordsInDoc = sum(wordCountList)
    tfidfVector = np.zeros(20000)
    for word, count in wordCountList:
        if word in topWordsDict:
            wordIndex = topWordsDict[word]
            idfValue = idfDict[wordIndex]
            tfidfValue = (count / nWordsInDoc) * idfValue
            tfidfVector[wordIndex] = tfidfValue
    return (docID, tfidfVector)

tfidfVectors = docWordDic.map(lambda doc: (doc[0], tfidfVectorize(doc)))
tfidfVectors.persist()


def normalizeVector(doc):
    docID = doc[0]
    tfidfVector = doc[1]
    mu = np.mean(tfidfVector)
    sigma = np.std(tfidfVector)
    normalizedVector = (tfidfVector - mu) / sigma if sigma != 0 else tfidfVector
    return (docID, normalizedVector)

tfidfNormalized = tfidfVectors.map(normalizeVector)
tfidfNormalized.persist()


def gradientDescentOptimized(r_init, lmd, lr, maxIter=30):
    r = r_init
    for i in range(maxIter):
        llhGrad = tfidfNormalized.map(lambda doc: (doc[0], calGrad(doc, r))).reduceByKey(lambda a, b: a + b).values().sum() / corpusSize
        r -= lr * (llhGrad + lmd * 2 * r) / corpusSize
    return r

rInit = np.array(r_from_small)
lmd = 0.0001
lr = 0.1
r = gradientDescentOptimized(rInit, lmd, lr, 100)

sortedIndices = np.argsort(r)
top50Idx = sortedIndices[-50:]
top50Words = [topWords[i][0] for i in reversed(top50Idx)]
print(top50Words)

# Additional code

for word in top50Words:
    print("Word: '{}', Coefficient: {}".format(word, r[topWordsDict[word]]))
