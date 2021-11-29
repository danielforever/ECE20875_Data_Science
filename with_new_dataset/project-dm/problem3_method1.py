import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.metrics import r2_score
clf = linear_model.LinearRegression()
from sklearn import metrics
import warnings
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
def makeDataset(wholedata, bridge,result):
    df  = pd.read_csv(wholedata,header=None,names=['Date',' Day','High Temp (°F)','Low Temp (°F)','Precipitation','Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge','Total'])
    df.replace(to_replace=['\s\(S\)'],value = '',inplace=True,regex=True)
    df.replace(to_replace=['T'],value = '0',inplace=True)
    df.replace(to_replace=[','],value = '',inplace=True,regex=True)
    
    df[result[0]] = df[result[0]][1:].astype(float)
    datasetx = []
    data=[]
    datasety=[]
    for i in bridge:
        for j in df[i][1:]:
            a=float(j)
            data.append(a)
            
        datasetx.append(data)
        data=[]
    for j in result:
        for i in df[j][1:]:
            if i!=0:
                a = 1
            else:
                a=0
            datasety.append(a)

    return datasetx,datasety


def accuraucy(coef,dataset,y,title):
    pre=[]
    ans=[]
    total=0
    dataset.append([float(1)]*len(dataset[0]))
    result = np.dot(np.array(coef),np.array(dataset))

    for i in range(len(dataset[0])):
        total+=((abs(result[i]-y[i]))/y[i])*100

    plt.title(title)
    x1,y1 = zip(*sorted(zip( range(len(dataset[0])),y)))
    plt.plot(x1,y1)
    x2,y2 = zip(*sorted(zip( range(len(dataset[0])), result)))
    plt.plot(x2,y2)
    plt.legend(['dataset','prediction'])
    plt.suptitle('',fontsize=10)
    plt.xlabel('day')
    plt.ylabel('number of riders')
    plt.show()
    
    return total/len(result)

def accuraucyforlinear(coef,dataset,y,title):
    pre=[]
    ans=[]
    total=0
    dataset.append([float(1)]*len(dataset[0]))
    result = np.dot(np.array(coef),np.array(dataset))
    for i in range(len(dataset[0])):
        if result[i] <=0.5:
            if result[i] <0:
                result[i]=0
            ansx=0
        else:
            ansx=1
        
        if y[i]>0:
            ansy=1

        else:
            ansy=0
        if ansx is ansy:
            total=total+1
    plt.title(title)
    x1,y1 = zip(*sorted(zip( range(len(dataset[0])),y)))
    plt.plot(x1,y1)
    x2,y2 = zip(*sorted(zip( range(len(dataset[0])), result)))
    plt.plot(x2,y2)
    plt.legend(['dataset','prediction'])
    plt.suptitle('',fontsize=10)
    plt.xlabel('day')
    plt.ylabel('Precipitation')
    print("accuraucy:",round(total/len(result),4))
    print("r:",r2_score(y,result))
    plt.show()
    plt.clf()
    return total/len(result)
    
if __name__ == '__main__':

    #only using Total on logistic function
    x,y	 = makeDataset('bike-data.csv', ['Total'],['Precipitation'])
    logreg = LogisticRegression()

    logreg.fit(np.array(x).transpose(),y)
    print(logreg.coef_)
    y_pred=logreg.predict(np.array(x).transpose())
    cnf_matrix = metrics.confusion_matrix(y, y_pred)
    print("Accuracy:",metrics.accuracy_score(y, y_pred))
    print("Precision:",metrics.precision_score(y, y_pred))
    print("Recall:",metrics.recall_score(y, y_pred))
    y_pred_proba = logreg.predict_proba(np.array(x).transpose())[::,1]
    fpr, tpr, _ = metrics.roc_curve(y,  y_pred_proba)
    auc = metrics.roc_auc_score(y, y_pred_proba)
    plt.plot(fpr,tpr,label="data, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

    #using all four bridges with total on logistic function
    x1,y1= makeDataset('bike-data.csv', ['Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge','Total'],['Precipitation'])
    logreg = LogisticRegression()

    logreg.fit(np.array(x1).transpose(),y1)
    print(logreg.coef_)
    y_pred1=logreg.predict(np.array(x1).transpose())
    cnf_matrix = metrics.confusion_matrix(y1, y_pred1)
    print("Accuracy:",metrics.accuracy_score(y1, y_pred1))
    print("Precision:",metrics.precision_score(y1, y_pred1))
    print("Recall:",metrics.recall_score(y1, y_pred1))
    y_pred_proba1 = logreg.predict_proba(np.array(x1).transpose())[::,1]
    fpr, tpr, _ = metrics.roc_curve(y1,  y_pred_proba1)
    auc = metrics.roc_auc_score(y1, y_pred_proba1)
    plt.plot(fpr,tpr,label="data, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

    #using linear function by the total number of riders
    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    
    print("coef:", coef)
    accuraucyforlinear(coef,x,y,'Precipitation')



