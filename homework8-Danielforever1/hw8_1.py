import os
import sys
import operator
import itertools

def getText(filename) :
    #fill in
    lines = []
##    print("filename", filename)
    with open(filename,'r') as f:
        an = f.readlines()
    return an  
###Arguments:
###  line: a string of text
###Returns: a list of n-grams
###Notes: make sure to pad the beginning and end of the string with '_'
###       make sure to convert the string to lower-case
###       so "Hello" should be turned into "__hello__" before processing
def getNgrams(line) :
    N=3
    line="__"+line.strip()+"__"
    line=line.lower()
    n_grams=[line[i:i+N] for i in range(len(line)- 2)]
    #print(n_grams)
    return n_grams
    
###Arguments:
###  filename: the filename to create an n-gram dictionary for
###Returns: a dictionary, with ngrams as keys, and frequency of that ngram as the value.
###Notes: Remember that getText gives you a list of lines, and you want the ngrams from
###       all the lines put together.
###       use 'map', use getText, and use getNgrams
###Hint: dict.fromkeys(l, 0) will initialize a dictionary with the keys in list l and an
###      initial value of 0
def getDict(filename):
    ans = []
    lines = getText(filename)
    for i in lines:
        ans.append(getNgrams(i))
    ans1 = list(set(itertools.chain.from_iterable(ans)))
    ans = list(itertools.chain.from_iterable(ans))
    #print(ans)
    dic = dict.fromkeys(ans1,0)
    for i in ans:
        dic[i] = dic[i]+1
    #print("dic",dic)
##    print(sorted(dic.items(),key=lambda x: x[1],reverse = True)[0:9])
    return sorted(dic.items(),key=lambda x: x[1],reverse = True)[0:9]
