import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from gmm_em import *
from sklearn.mixture import GaussianMixture


#function which computes a gaussian model with mean mu and variance var on the data array x
def gauss(x, mu, var):
    #fill in
    p = [ norm(mu,var**0.5).pdf(i) for i in x]
    return np.array(p)


#function which uses plt to plot the individual clusters and the full mixture model on a single chart
def plot_model(x, clusters, model):
    #fill in
##    print(len(clusters[1]))
    plt.plot(x,model, label='mixture',linestyle="dotted",linewidth=6)
    for i in range(len(clusters)):
        plt.plot(x, clusters[i],label='cluster'+str(i))
    plt.legend()
##    plt.show()
    name = "cluster_for_"+str(len(clusters))+".png"
    plt.savefig(name)
    plt.clf()
##    plt.show()


def main(weights, means, varis):
    #find range of inputted mixture model to be plotted
    [gmin, gmax] = [np.argmin(means), np.argmax(means)]
    xmin = means[gmin] - 4*np.sqrt(varis[gmin])
    xmax = means[gmax] + 4*np.sqrt(varis[gmax])

    #define range of 1000 points based on xmin and xmax
    inc = (xmax - xmin) / 1000
    x = np.arange(xmin,xmax+inc,inc)

    k = len(means)
    clusters = []   #a list of each component (gaussian) in the mixture applied to the vector x
    model = np.zeros(len(x))    #total mixture model applied to the vector x
    for i in range(k):
        p_i = gauss(x,means[i],varis[i])
##        print(p_i)
        clusters.append(weights[i]*p_i)
        model += weights[i]*p_i

    #plot the results
    plot_model(x,clusters,model)


##k=[2,3,4,5,6]
##for i in k:
##    weights,means,varis,ll = build_guass("data.txt",i,1)
##    main(weights, means, varis)
