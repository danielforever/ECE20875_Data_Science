import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import stats




dataset = [3, -3, 3, 12, 15, -16, 17, 19, 23, -24, 32]
t_c = t.ppf(1-(1-0.95)/2, len(dataset)-1)
print("\nproblem1")
print("S_score",t_c)
print("mean",np.mean(dataset))
print("stdError",np.std(dataset,ddof=1)/(len(dataset))**0.5)
print("the interval",np.mean(dataset)+(t_c*np.std(dataset,ddof=1)/(len(dataset))**0.5))
print("the interval",np.mean(dataset)-(t_c*np.std(dataset,ddof=1)/(len(dataset))**0.5))
print("\nproblem2")
t_c = t.ppf(1-(1-0.9)/2, len(dataset)-1)
print("\nS_score",t_c)
print("mean",np.mean(dataset))
print("stdError",np.std(dataset,ddof=1)/(len(dataset))**0.5)
print("the interval",np.mean(dataset)+(t_c*np.std(dataset,ddof=1)/(len(dataset))**0.5))
print("the interval",np.mean(dataset)-(t_c*np.std(dataset,ddof=1)/(len(dataset))**0.5))

print("\nproblem3")
t_c = t.ppf(1-(1-0.95)/2, len(dataset)-1)
print("\nS_score",t_c)
print("the interval",np.mean(dataset)+(t_c*16.836/(len(dataset))**0.5))
print("the interval",np.mean(dataset)-(t_c*16.836/(len(dataset))**0.5))
print("\nproblem4")

print("average",np.mean(dataset))
print("the interval",-(0-np.mean(dataset))*len(dataset)**0.5/np.std(dataset))
t_c = -(0-np.mean(dataset))*len(dataset)**0.5/np.std(dataset)
p = t.cdf(t_c, 11)
print(p)

