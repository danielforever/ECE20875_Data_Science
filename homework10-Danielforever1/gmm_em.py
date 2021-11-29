import numpy as np
from scipy.stats import norm
import math

#function which carries out the expectation step of expectation-maximization
def expectation(data, weights, means, varis):
    k = len(means)
    N = len(data)
    gammas = np.zeros((k,N))
    for i in range(len(means)):
        for j in range(len(data)):
            gammas[i][j] = (weights[i]*norm(means[i],varis[i]**0.5).pdf(data[j]))/sum([weights[ik]*norm(means[ik],varis[ik]**0.5).pdf(data[j])for ik in range(k)])
    #fill in here
    #code to calculate each gamma = gammas[i][j], the likelihood of datapoint j in gaussian i, from the
    #current weights, means, and varis of the gaussians
    return gammas


#function which carries out the maximization step of expectation-maximization
def maximization(data, gammas):
    k = len(gammas)
    N = len(data)
    weights = np.zeros(k)
    means = np.zeros(k)
    varis = np.zeros(k)
    for i in range(k):
        Ni = sum([gammas[i][j] for j in range(N)])
        weights[i] = Ni/N
        means[i] = sum([gammas[i][j]*data[j] for j in range(N)])/ Ni
        varis[i] = sum([gammas[i][j]*(data[j]-means[i])**2 for j in range(N)])/ Ni
        
    #fill in here
    #code to calculate each (i) weight = weights[i], the weight of gaussian i, (ii) mean = means[i], the
    #mean of gaussian i, and (iii) var = varis[i], the variance of gaussian i, from the current gammas of the
    #datapoints and gaussians

    return weights, means, varis


#function which trains a GMM with k clusters until expectation-maximization returns a change in log-likelihood of less
#than a tolerance tol
def train(data, k, tol):
    # fill in
    # initializations of gaussian weights, means, and variances according to the specifications
    weights = [1/k]*k
    varis = [1]*k
    means = [np.min(data) + i*(np.max(data)-np.min(data))/k for i in range(0, k)]

    diff = float("inf")
    ll_prev = -float("inf")
##
##    # iterate through expectation and maximization procedures until model convergence
    while(diff >= tol):
        gammas = expectation(data, weights, means, varis)
        weights, means, varis = maximization(data, gammas)
        
        ll = log_likelihood(data,weights,means,varis)
        diff = abs(ll - ll_prev)
        ll_prev = ll
##        print("here")
##    fml = "P(X) = "
##    for i in range(k):
##        fml+= str(round(weights[i],2))+"*norm("+str(round(means[i],2))+","+str(round(varis[i],2))+"**0.5).pdf(x)"
##        if i < k-1:
##            fml+="+"
##    print(fml)
    return weights, means, varis, ll


#calculate the log likelihood of the current dataset with respect to the current model
def log_likelihood(data, weights, means, varis):
    #fill in:
    ll = sum([math.log(sum([weights[i] * norm.pdf(data[j], means[i], math.sqrt(varis[i])) for i in range(len(means))])) for j in range(len(data))])
    
    return ll


def build_guass(datapath, k, tol):
    #read in dataset
    with open(datapath) as f:
        data = f.readlines()
    data = [float(x) for x in data]
##    data = [12.141406532590782, 4.55489200471575, 2.5673666260367822, 12.19351653969979, 12.78372755227871, 1.2055652005000008, 14.826872112706353, 4.643699755289818, 2.914079156255503, 1.4431893263445528, 14.73938738586368, 4.955765525134837, 13.009633670297589, 4.664401173563705, 4.5191443207949336]
    #train mixture model
    weights, means, varis, ll = train(data, k, tol)

    return weights,means,varis,ll
##k=[2,3,4,5,6]
##for i in k:
##    weights,means,varis,ll = build_guass("data.txt",i,1)
##    print("ll",ll)
