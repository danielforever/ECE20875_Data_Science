import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

clf = linear_model.LinearRegression()

#******************************************************************************  
    
#******************************************************************************
def makeDataset(wholedata, bridge,result,lower,upper):
    
    #make data frame
    df  = pd.read_csv(wholedata,header=None,names=['Date',' Day','High Temp (°F)','Low Temp (°F)','Precipitation','Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge','Total'])
    
    #Organize data
    df.replace(to_replace=['\s\(S\)'],value = '',inplace=True,regex=True)
    df.replace(to_replace=['T'],value = '0',inplace=True)
    df.replace(to_replace=[','],value = '',inplace=True,regex=True)

    #Convert data into float
    df[result[0]] = df[result[0]][1:].astype(float)
    
    
    #Get rid of the outliers
    #Upper bound
    indexname = df[df[result[0]]>int(upper)].index
    df.drop(indexname,inplace=True)
    
    #lower bound
    indexname = df[df[result[0]]<int(lower)].index
    df.drop(indexname,inplace=True)
    
    
    #make x matrix
    dataset_x = []
    data=[]
    for i in bridge:
        for j in df[i][1:]:
            a=float(j)
            data.append(a)
        dataset_x.append(data)
        data=[]
        
        
     #make y matrix
    dataset_y=[]
    for j in result:
        for i in df[j][1:]:
            a = i
            dataset_y.append(a)
            
    return dataset_x,dataset_y

#******************************************************************************  
    
#******************************************************************************
def accuraucy(coef,dataset,y,title):

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
    print("accuracy:",round(100-total/len(result),4))
    print("r:",r2_score(y,result))
    plt.show()
    
    
    return total/len(result)
    
        
#******************************************************************************  
    
#******************************************************************************
if __name__ == '__main__':
    
    #Brooklyn Bridge
    x,y	 = makeDataset('bike-data.csv', ['Precipitation','High Temp (°F)','Low Temp (°F)'],['Brooklyn Bridge'],1500,6000)
    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    print("coef:",coef)
    accuraucy(coef,x,y,'Brooklyn Bridge')


    #Manhattan Bridge
    x,y	 = makeDataset('bike-data.csv', ['Precipitation','High Temp (°F)','Low Temp (°F)'],['Manhattan Bridge'],1000,8000)
    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    print("coef:",coef)
    accuraucy(coef,x,y,'Manhattan Bridge')

    
    #Williamsburg Bridge
    x,y	 = makeDataset('bike-data.csv', ['Precipitation','High Temp (°F)','Low Temp (°F)'],['Williamsburg Bridge'],1000,7000)
    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    print("coef:",coef)
    accuraucy(coef,x,y,'Williamsburg Bridge')
    
    
    #Queensboro Bridge
    x,y	 = makeDataset('bike-data.csv', ['Precipitation','High Temp (°F)','Low Temp (°F)'],['Queensboro Bridge'],1000,8000)
    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    print("coef:",coef)
    accuraucy(coef,x,y,'Queensboro Bridge')
    
    
    #Total
    x,y	 = makeDataset('bike-data.csv', ['Precipitation','High Temp (°F)','Low Temp (°F)'],['Total'],6000,26000)
    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    print("coef:",coef)
    accuraucy(coef,x,y,'Total')
