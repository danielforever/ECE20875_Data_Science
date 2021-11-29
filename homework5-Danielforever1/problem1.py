import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import stats


myFile = open('eng0.txt')
data = myFile.readlines()
myFile.close()
myFile2 = open('eng1.txt')
data2 = myFile2.readlines()
myFile2.close()
##print(data)
data = [float(x) for x in data]
print("data size:",len(data))
data2 = [float(x) for x in data2]
print("data2 size:",len(data2))
##[print(i) for i in data]
print("data mean:",np.mean(data))
print("data2 mean:",np.mean(data2))

#var
print("data2 var",np.var(data2))
std_b = np.std(data2,ddof=1)
##s = np.sqrt(var_a/len(data)+ var_b/len(data2))
s = (np.mean(data)*len(data) + np.mean(data2)*len(data2))/(len(data2)+len(data))
print("s",s)
n=np.sqrt(len(data2))
print("data2 SE:" , stats.sem(data2))
print("z-score",(np.mean(data2)-0.75)/(stats.sem(data2)))
z = (np.mean(data2)-0.75)/(std_b/len(data2)**0.5)
z_c = norm.ppf(0.05)
print("z",z_c)
n = (std_b*z_c/(np.mean(data2)-0.75))**2
print("n:",n)
print("p-value",2*stats.norm.cdf(z))

var_a = np.var(data)
var_b = np.var(data2)
s_com = (((var_a)/len(data))+((var_b)/len(data2)))**0.5
print("s_com",s_com)
z_score_com = ((np.mean(data2)-np.mean(data))-0.75)/s_com
print("z_score_com",z_score_com)
print("p-value",2*stats.norm.cdf(z_score_com))
##avg = np.mean(data)
##print("data Std:",np.std(data,ddof=1))#stand deviation
##print("data2 Std:",np.std(data2,ddof=1))
#standard deviation


##print("data SE:" , stats.sem(data))
##print("data zscore", stats.zscore(data))#axis n-1 degrees2

##print("data2 zscore", stats.zscore(data2))#axis n-1 degrees
##print("data zscore", stats.zscore(data,axis=-1,ddof=1))#axis n-1 degrees
