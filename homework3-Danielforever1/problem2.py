import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from helper import *

#def getData(filename):
#    file = open(filename,'r')
#    return file.read()

data = getData('distA.csv')
stats.probplot(data, dist = stats.uniform, plot=plt)
plt.savefig('distA-uniform.png')
plt.show()
data = getData('distB.csv')
stats.probplot(data, dist = stats.expon, plot=plt)
plt.savefig('distB-expon.png')
plt.show()
data = getData('distC.csv')
stats.probplot(data, dist = 'norm', plot=plt)
plt.savefig('distC-norm.png')
plt.show() # modify this to write the plot to a file instead
#stats.probplot(data, dist = stats.cauchy, plot=plt)
#plt.savefig('distB-cauchy.png')
#plt.show()
#stats.probplot(data, dist = stats.cosine, plot=plt)
#plt.savefig('distB-cauchy.png')
#plt.show()
#stats.probplot(data, dist = stats.expon, plot=plt)
#plt.savefig('distB-expon.png')
#plt.show()
#stats.probplot(data, dist = stats.uniform, plot=plt)
#plt.savefig('distB-uniform.png')
#plt.show()
#stats.probplot(data, dist = stats.laplace, plot=plt)
#plt.savefig('distA-lapalce.png')
#plt.show()
#stats.probplot(data, dist = stats.wald, plot=plt)
#plt.savefig('distB-cauchy.png')
#plt.show()
#stats.probplot(data, dist = stats.rayleigh, plot=plt)
#plt.savefig('distB-cauchy.png')
#plt.show()

