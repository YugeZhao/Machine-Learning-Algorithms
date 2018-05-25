# -*- coding: utf-8 -*-
import os;
import sys;
import glob;
import csv;
import fractions;
import math;
import matplotlib.pyplot as plt;
import numpy as np;
#print(np.__path__);
import pandas as pd;
import scipy.stats;
import matplotlib;
#print(scipy.__path__);
from scipy.optimize import minimize;
from scipy.stats._multivariate import multivariate_normal;
from numpy.core.umath_tests import matrix_multiply;
from fractions import Fraction
from idlelib.ReplaceDialog import replace
from scipy.constants.constants import sigma
from setuptools.dist import Feature
from matplotlib.pyplot import plot, axis
from pip._vendor.distlib.util import CSVReader
from asyncore import read
from StdSuites.AppleScript_Suite import string
from numpy import matrix
from Finder.Type_Definitions import column
from StdSuites.Table_Suite import row

##Static Expert algorithm for online learing
#Input: d the number of features (columns of the data), and learning rate b, t is the # of iterations
#Output

def staticExpert(featureData, labelData, d, b, t):
    #initialize
    t0 = 1;
    nx,mx = featureData.shape; #number of data points
    ny,my = labelData.shape;
    Loss = np.zeros(len(featureData));
    Pt = np.zeros(d); 
    Yt = np.zeros(d);
      
                
    while (t0 <= t):
        np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())});
        for i in range(d):
            Pt[i] = Fraction(1,d);
        for row1, row2 in featureData, labelData:
            ##Output prediction
            for l in d:
                y0 = 0;
                yt = Pt[l]*row1[l];
                y0 = y0 + yt; 
                
            Loss[row1] = (y0 - row2[l])**2;
            plot(t0, Loss[row1]);
            Pt[i] = Pt[i]*(math.exp(-b*Loss[row1]));
        t = t+1;
            
def to2dMatrix(reader):
    entry = [];
    #i = 53; it = 2105;
    for row in reader:
        entry.append(row);
    return entry

def column(matrix, i):
    return [row[i] for row in matrix]

with open('cloud.data') as Data:
    tempData = [];
    cloudData = [];
    cloudLabel = [];
    for row in Data:
        tempData.append(row);
    for r in range(54, 1078):
        cloudData.append(tempData[r]);
    for k in range(1082, 2107):
        cloudData.append(tempData[k]);
    cloudData = np.matrix(cloudData).T;
    #for row in cloudData:
     #   row[0,0].split();
        #print(row);

    print(cloudData);print(cloudData.shape);

    cloudLabel  = column(cloudData, 6);
    #print(cloudLabel);
    #cloudData = np.matrix(cloudData);
d = 9; b1 = 1; t = 100;
staticExpert(cloudData, cloudLabel, d, b1, t);
b2 = 2;
staticExpert(cloudData, cloudLabel, d, b2, t);
b3 = 3;
staticExpert(cloudData, cloudLabel, d, b3, t);

         