from helper import remove_punc
import io
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
#Clean and lemmatize the contents of a document
#Takes in a file name to read in and clean
#Return a list of words, without stopwords and punctuation, and with all words lemmatized
# NOTE: Do not append any directory names to doc -- assume we will give you
# a string representing a file name that will open correctly
def readAndCleanDoc(doc) :
    no_stop=[]
    words=[]
    #1. Open document, read text into *single* string
    with open(doc,'r') as f:
        allfile = f.read()
    #2. Tokenize string using nltk.tokenize.word_tokenize
    word_toke = word_tokenize(allfile)
    
    #3. Filter out punctuation from list of words (use remove_punc)
    word_toke = remove_punc(word_toke)
    #4. Make the words lower case
    word_toke = [i.lower() for i in word_toke]
    #5. Filter out stopwords
##    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
##    print(stop_words)
    for i in word_toke:
        if not i in stop_words:
            no_stop.append(i)
##    print(no_stop)
    #6. Lemmatize words
    lemmatizer = WordNetLemmatizer()
    for i in no_stop:
        words.append(lemmatizer.lemmatize(i))
    return words
    
#Builds a doc-word matrix for a set of documents
#Takes in a *list of filenames*
#
#Returns 1) a doc-word matrix for the cleaned documents
#This should be a 2-dimensional numpy array, with one row per document and one 
#column per word (there should be as many columns as unique words that appear
#across *all* documents
#
#Also returns 2) a list of words that should correspond to the columns in
#docword
#list(set(itertools.chain.from_iterable(ans)))
def buildDocWordMatrix(doclist) :
    words_all_file = []
    doc_word = []
    word_vec = []
    
    #1. Create word lists for each cleaned doc (use readAndCleanDoc)
    for i in doclist:
        words_all_file.append(readAndCleanDoc(i))
    for words in words_all_file:
        for word in words:
            if(word not in word_vec):
                word_vec.append(word)
##    print(word_vec)
##    for i in word_vec:
##        print(len(word_vec))
    for file in words_all_file:
        file_vec = [0]*len(word_vec)
        for i in file:
            file_vec[word_vec.index(i)] =file_vec[word_vec.index(i)]+1
        doc_word.append(file_vec)
    docword = np.array(doc_word)
##    print(docword)
    wordlist = np.array(word_vec)
    #2. Use these word lists to build the doc word matrix 
    return docword, wordlist
##    
###Builds a term-frequency matrix
###Takes in a doc word matrix (as built in buildDocWordMatrix)
###Returns a term-frequency matrix, which should be a 2-dimensional numpy array
###with the same shape as docword
def buildTFMatrix(docword) :
##    print(docword)
    #fill in
    tf=[]
    for i,j in enumerate(docword):
        tf.append(j/np.sum(j))
    tf=np.array(tf)
    return tf
##    
###Builds an inverse document frequency matrix
###Takes in a doc word matrix (as built in buildDocWordMatrix)
###Returns an inverse document frequency matrix (should be a 1xW numpy array where
###W is the number of words in the doc word matrix)
###Don't forget the log factor!
def buildIDFMatrix(docword) :
    trans = docword.T
    #fill in
    idf = []
##    print("idf",idf)
    log_con = len(trans[0])
##    print("log",log_con)
    for i in trans:
        total = 0
        for j in i:
            if j!=0:
                total+=1
##            print(np.log10(log_con/total))
##        if total!=0:
        idf.append(np.log10(log_con/total))
##        else:
##            idf.append(1)
##    print("martix??",np.asmatrix([idf]))
##    print(np.asmatrix([idf]).shape)
    return np.array([idf])

##    
###Builds a tf-idf matrix given a doc word matrix
def buildTFIDFMatrix(docword) :
    return buildTFMatrix(docword)*buildIDFMatrix(docword)
##    
###Find the three most distinctive words, according to TFIDF, in each document
###Input: a docword matrix, a wordlist (corresponding to columns) and a doclist 
### (corresponding to rows)
###Output: a dictionary, mapping each document name from doclist to an (ordered
### list of the three most common words in each document
def findDistinctiveWords(docword, wordlist, doclist) :
    distinctiveWords = {}
    tfidf = buildTFIDFMatrix(docword)
    for ind,file in enumerate(doclist):
        top=[wordlist[ind] for ind in np.argsort(tfidf[ind])[-3:]]
        distinctiveWords.update({file:sorted(top)})
    #fill in
    #you might find numpy.argsort helpful for solving this problem:
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    
    return distinctiveWords
if __name__=="__main__":
    print("*** Testing readAndCleanDoc ***")
    print(readAndCleanDoc('lecs/1_vidText.txt')[0:5])

    print("*** Testing buildDocWordMatrix ***")
    doclist = ['lecs/1_vidText.txt', 'lecs/2_vidText.txt']
    docword, wordlist = buildDocWordMatrix(doclist)
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])

    print("*** Testing buildTFMatrix ***")
    tf = buildTFMatrix(docword)
    print(tf[0][0:10])
    print(tf[1][0:10])
    print(tf.sum(axis=1))

    print("*** Testing buildIDFMatrix ***")
    idf = buildIDFMatrix(docword)
    print(idf[0][0:10])

    print("*** Testing buildTFIDFMatrix ***")
    tfidf = buildTFIDFMatrix(docword)
    print(tfidf.shape)
    print(tfidf[0][0:10])
    print(tfidf[1][0:10])

    print("*** Testing findDistinctiveWords ***")
    print(findDistinctiveWords(docword, wordlist, doclist))
