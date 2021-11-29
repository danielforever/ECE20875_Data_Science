import numpy as np
import matplotlib.pyplot as plt
from helper import *

#change a histogram of counts into a histogram of probabilities
#input: a histogram (like your histogram function creates)
#output: a normalized histogram with probabilities instead of counts
def norm_histogram(hist) :
    #fill in
    #hint: when doing your normalization, you should convert the integers
    #      in your histogram buckets to floats: float(x)
    norm_hist=list()
    #norm_hist.append(i/sum(hist)) for i in hist
    for i in hist:
        norm_hist.append((i/sum(hist)))        
    return norm_hist
    
#compute cross validation for one bin width
#input: a (non-normalized) histogram and a bin width
#output: the cross validation score (J) for the specified width
def crossValid (histo, width) :
    #1. build the list of probabilities
    #print(histo,"histo")
    #print(width)
    #print(histo)
    norm_hist = norm_histogram(histo)
    J = [0]*len(histo)
    #print(norm_hist)
    #2. compute the sum of the squares of the probabilities
    sum_of_squares = float()
    for i in norm_hist:
        sum_of_squares += i**2
    #print(sum_of_squares,"check")
    #3. determine the total number of points in the histogram
    #   hint: look up the Python "sum" function
    total_points = sum(histo)
    #print(total_points)
    #4. Compute J(h) and return it
    #print(norm_hist)
    #print(len(histo),"here")
    J = (2/((total_points-1)*width))-((total_points+1)/((total_points-1)*width))*sum_of_squares
    #for i in range(0,len(histo)):
    #    J[i] = (2/((histo[i]-1)*width))-((histo[i]+1)/((histo[i]-1)*width))*sum_of_squares
    #print(J)
    return J
    
#sweep through the range [min_bins, max_bins], compute the cross validation
#score for each number of bins, and return a list of all the Js
#Note that the range is inclusive on both sides!
#input: the dataset to build a histogram from
#       the minimum value in the data set
#       the maximum value in the data set
#       the smallest number of bins to test
#       the largest number of bins to test
#output: a list (of length max_bins - min_bins + 1) of all the appropriate js
def sweepCross (data, minimum, maximum, min_bins, max_bins) :
    #fill in. Don't forget to convert from a number of bins to a width!
    if (len(data)<=0 or maximum < minimum or isinstance(len(data), int) is False):
        print("n is not good")
        return []
    #print(len(data),"len")
    js = list()
    for j in range(min_bins, max_bins+1):
        #print(j)
        w = (maximum-minimum)/j
        histo = [0]*j
        for i in data:
            if (i >= minimum and i <= maximum):
                bin = (i-minimum)/w
                if i == maximum:
                    histo[int(bin)-1]=histo[int(bin)-1]+1
                else:
                    histo[int(bin)]=histo[int(bin)]+1
        js.append(crossValid(histo,w))
    return js
        
#return the minimum value in a list *and* the index of that value
#input: a list of numbers
#output: a tuple with the first element being the minimum value, and the second 
#        element being the index of that minumum value (if the minimum value is 
#        in the list more than once, the index should be the *first* occurence 
#         of that minimum value)
def findMin (l) :
    #fill in.
    minVal = min(l)
    minIndex = l.index(minVal)
    return (minVal, minIndex)

        
if __name__ == '__main__' :
        #Sample test code
        data = getData() #reads data from inp.txt
        lo = min(data)
        hi = max(data)
        #h = histogram(data, 10, lo, hi)
        #n_h = norm_histogram(h)
        #print(n_h)
        #width = (hi - lo) / 10
        #print(width)
        #print(crossValid(h, width))
        js = sweepCross(data, lo, hi, 1, 100)
        #print(js)
        #js = sweepCross(data, lo, hi, 1, 5)
        #print(js)
        
        print(findMin(js))
