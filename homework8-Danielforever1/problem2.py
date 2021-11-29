#Arguments:
#  filename: name of file to read in
#Returns: a list of strings, each string is one line in the file
#hints: https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
#       https://docs.python.org/3/library/stdtypes.html#str.splitlines
import sys
from sys import exit
import itertools
import operator
import helper
import numpy as np
import matplotlib.pyplot as plt
import string
from helper import *

def plotHisto_(bars, filename, minrange , maxrange, plotinline ) :
	mrange = maxrange - minrange
	binsize = mrange/len(bars)
	
	#this is a "list comprehension" -- it's a quick way to process one
	#list to produce another list
	labels = [(mrange / len(bars)) * i + minrange for i in range(len(bars))]
##	print("bars:",bars)
	plt.bar(labels, bars, align = 'edge', width = binsize)
	plt.title(filename)
	if plotinline :
		plt.show()
	else :
		plt.savefig(filename)
##		plt.show()
		plt.clf()

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
    bar = []
    for i in filename:
        lines = getText(i)
        for j in lines:
            ans.append(getNgrams(j))
    ans1 = list(set(itertools.chain.from_iterable(ans)))
##    print("ans1:",len(ans1))
##    filename = ['ngrams/mystery.txt']
    for i in filename:
        dic = dict.fromkeys(ans1,0)
        bar = []
        anseach = []
        lines = getText(i)
        name = i.split('/')[1]
        for j in lines:
            anseach.append(getNgrams(j))
        anseach = list(itertools.chain.from_iterable(anseach))
        for j in anseach:
            if j in dic.keys():
                dic[j] +=1
##        print("value:",sum(dic.values()))
##        print(sorted(dic.keys()))
##        print(sorted(dic.keys()))
        for j in sorted(dic.keys()):
            bar.append(dic[j])
##        print("Bar",bar)
##        print(sorted(dic.items(),key=lambda x: x[1],reverse = True)[0:10])
##        print(bar)
##        print("startHisto")
##        plotHisto(bar,name+".png",0.0, len(dic.keys()),False)
        plotHisto(bar,name+".png",0.0, len(dic.keys()),False)
##        print("endHisto")
    return sorted(dic.items(),key=lambda x: x[1])





english_ngrams = getDict(['ngrams/english.txt','ngrams/french.txt','ngrams/german.txt','ngrams/italian.txt','ngrams/portuguese.txt','ngrams/spanish.txt'])
print(english_ngrams)
##english_ngrams = getDict(['ngrams/english.txt'])
##french_ngrams = getDict(['ngrams/french.txt'])
##german_ngrams = getDict(['ngrams/german.txt'])
##italian_ngrams = getDict(['ngrams/italian.txt'])
##portuguese_ngrams = getDict(['ngrams/portuguese.txt'])
##spanish_ngrams = getDict(['ngrams/spanish.txt'])
##mistery_ngrams = getDict(['ngrams/mystery.txt'])
