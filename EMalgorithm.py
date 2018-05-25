# -*- coding: utf-8 -*-
import os;
import sys;
import glob;
import csv;
import fractions;
import matplotlib.pyplot as plt;
import numpy as np;
#print(np.__path__);
import pandas as pd;
import scipy.stats;
#print(scipy.__path__);
from scipy.optimize import minimize;
from scipy.stats._multivariate import multivariate_normal;
from numpy.core.umath_tests import matrix_multiply;
from fractions import Fraction
from idlelib.ReplaceDialog import replace
from scipy.constants.constants import sigma

###Mixture Gaussian distribution using EM algorithm. 
#The function estimates a mixture of k spherical Gaussians with y observations
#z denotes the latent variables 
#ωj,µj denote the mixing proportions, means of each Gaussian component y = j
#xi denote a data point, k denote the number of Gaussian components,
#N denote the number of data points, and nj denote the effective number of data points currently assigned to  component j
#Initialize parameters theta0 at random
#at E-step, For each data point xi, for each component j: update P(y = j|()xi)
#at M-step,  Find new parameters theta by maximizing the complete log-likeihood
def EM(x, k, omega, mu, sigma, maxIteration, tolerance=0.01):
    #k = len(omega);
    n,m  = x.shape; #dimension of x (n,m)
    loglike0 = 0;
    for l in range(maxIteration):
        expectA = [];
        expectB = [];
        loglikeN = 0;
        #E-step
        P = np.zeros((k,n));
        for j in range(k):
            #MVNj = np.random.multivariate_normal(mu[j], sigma[j]);
            P[j,:] = omega[j]*multivariate_normal(mu[j], sigma[j]).pdf(x);
        P /= P.sum(0);
        
        #M-steo
        omega = P.sum(axis=1);
        omega /= n;
        mu = np.dot(P,x);
        mu /= P.sum(1)[:,None];
        sigma = np.zeros((k,m,m));
        for j in range(k):
            y = x - mu[j,:];
            sigma[j] = (P[j,:,None,None]*matrix_multiply(y[:,:,None], y[:,None,:])).sum(axis=0);
        sigma /= P.sum(axis=1)[:,None,None];
        
        #Update complete log likelihood 
        loglikeN = 0;
        for omega, mu, sigma in zip(omega,mu,sigma):
            #MVN = np.random.multivariate_normal(mu,sigma);
            loglikeN += omega*multivariate_normal(mu,sigma).pdf(x);
        loglikeN = np.log(loglikeN).sum();
        if np.abs(loglikeN - loglike0) < tolerance:
            break
        loglike0 = loglikeN;

    return loglikeN, omega, mu, sigma;

def to2dMatrix(reader):
    entry = []
    for row in reader:
        if row == "":  break
        else:   pass
        if row[-1] == '\n':
            row = row[:-1]
        else:   pass
        entry.append(row)
    return entry

####Problem(e)(i) Initialization
with open('irisData.csv','rb') as irisData:
    readerData = csv.reader(irisData);
    Data = to2dMatrix(readerData);
    Data = np.matrix(Data);
    print(Data);print(Data.shape);
with open('irisLabels.csv','rb')as irisLabels:
    readerLabels = csv.reader(irisLabels);
    Labels = to2dMatrix(readerLabels);
    Labels = np.matrix(Labels);
    #print(Labels);
#trainData = np.concatenate(Data,Labels);
#print(trainData);
np.random.seed(1234);
####Problem 1.2.Run EM algorithm with k = 2,  iterations = 10,100

def Train(k, iterations, Data):
    np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())});
    #np.set_printoptions(formatter={'all':lambda x: '%.3f' % x});
    if k==2:
        omega = np.matrix('1.0 1.0');
        omega[0,0] = Fraction(1,k);
        omega[0,1] = Fraction(1,k);
    if k==3:
        omega = np.matrix('1.0 1.0 1.0');
        omega[0,0] = Fraction(1,k);
        omega[0,1] = Fraction(1,k);
        omega[0,2] = Fraction(1,k);
    #sigma1 = np.matrix('1.0,0,0,0; 0,1.0,0,0; 0,0,1.0,0; 0,0,0,1.0');
    sigma = np.matrix([np.eye(4)]*k);
    print(omega);print(omega.shape);print(sigma);print(sigma.shape);
    mu = Data[np.random.choice(Data.shape[0],2,replace=False)];
    mu = mu.T;
    print(mu);print(mu.shape);
    return EM(Data, k, omega, mu, sigma, iterations);

k1 = 2;
iterations1 =  10100;
#Train(k1, iterations1, Data)

k2 = 2;
iterations2 = 1000;
#Train(k2, iterations2, Data);

k3 = 3;
iterations3 = 10100;
#Train(k3, iterations3, Data);

###Problem 3.
mu1 = np.matrix('1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0');
#Train(k2, iterations2, Data);

mu2 = np.matrix('1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0');
#Train(k3, iterations2, Data);

                     