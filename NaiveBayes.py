import csv
import random
import math 
import pandas as pd
import numpy as np
from twisted.python.test.test_reflectpy3 import Separate
from cProfile import label
from sklearn.svm.libsvm import predict
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from lib2to3.pgen2.tokenize import group

def load(file):
    line = csv.reader(open(file,"rb"))
    data = list(line)
    for i in range(len(data)):
        data[i] = [float(x) for x in data[i]]
    return data

def group(TrainData,TrainLabel):
    groups = {}
    for i in range(len(TrainData)):
        group = TrainLabel[i]
        if(group[0] not in groups):
            groups[group[0]] = []
        groups[group[0]].append(TrainData[i])       
    return groups

def count1(col):
    count1 = 0
    for i in range (len(col)):
        if (col[i]==1):
            count1 = count1+1
    return count1

def countTotal(col):
    total = len(col)
    return total

def evaluate(data):
    groups = zip(*data)
    evaluate = [(count1(attr),countTotal(attr)) for attr in groups]
    return evaluate

def evaluateGroups(TrainData,TrainLabel):
    groups = group(TrainData, TrainLabel)
    evaluations = {}
    for label, instances in groups.iteritems():
        evaluations[label] = evaluate(instances)
    return evaluations
    
def Prob(xi,count1,total):
    pi = smooth(count1, total)
    x = math.pow(float(pi), float(xi))
    y = math.pow(1.0-pi, 1.0-xi)
    prob = x*y
    return prob

def smooth(count1,total):
    n = 2
    ps = 0.5
    pi = float((count1+n*ps)/(total+n))
    return pi
  
def NBProb(evaluations,data):
    prob = {}
    for label, instances in evaluations.iteritems():
        prob[label] = 1
        for i in range(len(instances)):
            count1, total = instances[i]
            xi = data[i]
            prob[label] *=  Prob(xi,count1,total)
    return prob

def NBpred(evaluations, data):
    prob = NBProb(evaluations, data)
    label0,prob0 = None, -1
    for val, probs in prob.iteritems():
        if label0 is None or probs>prob0:
            prob0 = probs
            label0 = val
    return label0

def NBpredicts(evaluations, TestData):
    predNB = []
    for i in range(len(TestData)):
        predictions = NBpred(evaluations, TestData[i])
        predNB.append(predictions)
    return predNB

def error(TestLabel, pred):
    rate = 0 
    for i in range(len(TestLabel)):
        if TestLabel[i][0] != pred[i]:
            rate += 1
    return (rate/float(len(TestLabel)))*100.0
                
fileRD = '/Users/annazhao/eclipse-workspace/Machine Learning HW4/src/SpectTrainData.csv'
TrainData = load(fileRD)
fileRL = '/Users/annazhao/eclipse-workspace/Machine Learning HW4/src/SpectTrainLabels.csv'
TrainLabel = load(fileRL)
fileTD = '/Users/annazhao/eclipse-workspace/Machine Learning HW4/src/SpectTestData.csv'
TestData = load(fileTD)
fileTL = '/Users/annazhao/eclipse-workspace/Machine Learning HW4/src/SpectTestLabels.csv'
TestLabel = load(fileTL)

groups = group(TrainData,TrainLabel)
evaluations = evaluateGroups(TrainData, TrainLabel)
prediction = NBpredicts(evaluations, TestData)
error = error(TestLabel, prediction)

print('Error rate: {0}%').format(error)


