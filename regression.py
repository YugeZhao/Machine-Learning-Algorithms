import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import regression
from matplotlib.pylab import rcParams
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from numpy import size, shape, nonzero, append, matrix
import test
from pandas.core.frame import DataFrame

def ridge_regression(trainX, trainY, alpha, testX, testY):
    #Problem (a) (b) (e)
    ridgeReg = Ridge(alpha);
    ridgeReg.fit(trainX,trainY);
    pred = ridgeReg.predict(testX);
    plt.plot(ridgeReg.alpha)

    nonZero = []
    for p in ridgeReg.coef_:
        if any(p!=0): 
            nonZero.append(p)
    count = size(nonZero)
    print(alpha)
    print(nonZero)
    print(count)
    
    #Problem (c) (d)
    rss = sum((pred-testY)**2)
    ret = [rss]
    ret.extend([ridgeReg.intercept_])
    ret.extend(ridgeReg.coef_)
    return ret

def lasso_regression(trainX, trainY, alpha, testX, testY):
    #Fit the model
    lassoReg = Lasso(alpha);
    lassoReg.fit(trainX,trainY);
    pred = lassoReg.predict(testX);
    lassoReg.score(testX,testY)
    #print(lassoReg.coef_);
    nonZero = []
    for i in range(19):
        if lassoReg.coef_[i]!= 0:
            #print(lassoReg.coef_[i])
            plt.plot(lassoReg.alpha)
            nonZero.append(lassoReg.coef_[i])
    count = size(nonZero)
    print(alpha)
    print(nonZero)
    print(count)
    
    #Return the result in pre-defined format
    rss = sum((pred-testY)**2)
    #print(rss)
    plt.plot(rss)
    ret = [rss]
    ret.extend([lassoReg.intercept_])
    ret.extend(lassoReg.coef_)
    return ret

########Data set A: randomly simulated data set created as follows

trainX = np.random.normal(size =[1000, 20])
trainY = np.random.normal(size =[1000, 1])
testX = np.random.normal(size =[500, 20])
testY = np.random.normal(size =[500, 1])

####Problem 1: LASSO Regression

alpha1 = 0.01;
alpha2 = 0.05;
alpha3 = 0.1;
alpha4 = 0.2;
alpha5 = 0.3;
lasso_regression(trainX, trainY, alpha1, testX, testY);
lasso_regression(trainX, trainY, alpha2, testX, testY);
lasso_regression(trainX, trainY, alpha3, testX, testY);
lasso_regression(trainX, trainY, alpha4, testX, testY);
lasso_regression(trainX, trainY, alpha5, testX, testY);
####Answers to (a)(b)(c): in this data set, wb is 0.1, 0.2,0.3 which all yield
#### 0 non zero coefficients
#####Problem 2: Ridge Regression
ridge_regression(trainX, trainY, alpha1, testX, testY);
ridge_regression(trainX, trainY, alpha2, testX, testY);
ridge_regression(trainX, trainY, alpha3, testX, testY);
ridge_regression(trainX, trainY, alpha4, testX, testY);
ridge_regression(trainX, trainY, alpha5, testX, testY);

########Data set B: Cloud Data
with open('cloud.data') as Data:
    tempData = [];
    cloud = [];
    cloudData = [];
    cloudlabel = [];
    test = [];
    label = [];
    for row in Data:
        tempData.append(row);
    for r in range(54, 1078):
        cloud.append(tempData[r]);
    for k in range(1082, 2107):
        cloud.append(tempData[k]);
    for i in range(10):
        cloud[i] = cloud[i].split(' ')
        cloud[i].remove('')
        del cloud[i][0];del cloud[i][1];del cloud[i][2];del cloud[i][5]
        cloud[i].remove('')
        del cloud[i][2];del cloud[i][3];del cloud[i][4]
        cloud[i].remove('');del cloud[i][2]
        cloud[i].remove('');cloud[i].remove('');cloud[i].remove('')
        cloud[i].remove('');cloud[i].remove('');cloud[i].remove('')
        cloud[i].remove('');cloud[i].remove('');cloud[i].remove('')
        cloud[i].remove('');
        del cloud[i][8]
    for j in range(1023):
        label.append(cloud[j][6])
    for n in range(799):
        cloudData.append(cloud[n])
    for h in range(799):
        cloudlabel.append(label[h])
    for a in range(800,size(cloud)):
        test.append(cloud[a])
    for b in range(800,size(label)):
        test.append(label[b])
    #print(cloud) #2049
    #print(label)
    #print(cloudData)
    #print(cloudlabel)
    
####Problem B1: LASSO Regression
#lasso_regression(cloudData, cloudlabel, alpha1, test, label);
#lasso_regression(cloudData, cloudlabel, alpha2, test, label);
#lasso_regression(cloudData, cloudlabel, alpha3, test, label);
#lasso_regression(cloudData, cloudlabel, alpha4, test, label);
#lasso_regression(cloudData, cloudlabel, alpha5, test, label);

####Problem B2: Ridge Regression
#ridge_regression(cloudData, cloudlabel, alpha1, test, label);
#ridge_regression(cloudData, cloudlabel, alpha2, test, label);
#ridge_regression(cloudData, cloudlabel, alpha3, test, label);
#ridge_regression(cloudData, cloudlabel, alpha4, test, label);
#ridge_regression(cloudData, cloudlabel, alpha5, test, label);

#########Data set 3: Forest
with open('forestfires.csv') as data:
    #tempData = [];
    forest = [];
    forestData = [];
    forestlabel = [];
    test1 = [];
    label1 = [];
    for row in data:
        forest.append(row);
    for j in range(399):
        forestlabel.append(forest[j][12])
    for n in range(399):
        forestData.append(forest[n])
    for h in range(400,517):
        label1.append(forest[h])
    for a in range(400,517):
       
        test1.append(forest[a])
        
####Problem B1: LASSO Regression
#lasso_regression(forestData, forestlabel, alpha1, test1, label1);
#lasso_regression(forestData, forestlabel, alpha2, test1, label1);
#lasso_regression(forestData, forestlabel, alpha3, test1, label1);
#lasso_regression(forestData, forestlabel, alpha4, test1, label1);
#lasso_regression(forestData, forestlabel, alpha5, test1, label1);

####Problem B2: Ridge Regression
#ridge_regression(forestData, forestlabel, alpha1, test1, label1);
#ridge_regression(forestData, forestlabel, alpha2, test1, label1);
#ridge_regression(forestData, forestlabel, alpha3, test1, label1);
#ridge_regression(forestData, forestlabel, alpha4, test1, label1);
#ridge_regression(forestData, forestlabel, alpha5, test1, label1);
