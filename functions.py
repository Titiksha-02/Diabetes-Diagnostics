# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 22:29:29 2021

@author: Titiksha
Roll No.: B20138
Mobile Number: 9811162646
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def replace_out(data,column) :  # identify outliers and replace them with median
    A={}
    for i in range(8) :
        arr=data[column[i]]
        q1=np.percentile(arr,25,interpolation = 'midpoint')        
        q3=np.percentile(arr,75,interpolation = 'midpoint') 
        q2=np.median(arr)
        IQR=q3-q1
        for j in arr.index :
            if arr[j]>(q3+(1.5*IQR)) or arr[j]<(q1-(1.5*IQR)) :
                arr=arr.replace(arr[j],q2)
        A[column[i]]=arr
    data2=pd.DataFrame(A,columns=column[:8])
    return data2

def BA_nor(df1,df2,column): # finding max, min before and after normalisation
    maxB=[]
    minB=[]
    maxA=[]
    minA=[]
    for i in range(8):
        maxB.append(max(df1[column[i]]))
        minB.append(min(df1[column[i]]))
        maxA.append(max(df2[column[i]]))
        minA.append(min(df2[column[i]]))
    df=pd.DataFrame(list(zip(column[:8],minB,maxB,minA,maxA)),
                    columns=['Attribute','Min Before','Max Before','Min After','Max After'])
    return df

def normal(arr) : # normalise the data in the range 5-12
    Max=max(arr)
    Min=min(arr)
    new_max=12
    new_min=5
    m=(new_max-new_min)/(Max-Min)
    arr2=[]
    for i in arr.index :
        arr2.append(((arr[i]-Min)*m)+new_min)
    return arr2


def BA_std(df1,df2,column) : # finding mean, std before and after standardisation
    meanB=[]
    stdB=[]
    meanA=[]
    stdA=[]
    for i in range(8) :
        meanB.append(round(np.mean(df1[column[i]]),5))
        stdB.append(round(np.std(df1[column[i]]),5))
        meanA.append(round(np.mean(df2[column[i]]),5))
        stdA.append(round(np.std(df2[column[i]]),5))
    df=pd.DataFrame(list(zip(column[:8],meanB,stdB,meanA,stdA)),
                    columns=['Attribute','Mean Before','Std Before','Mean After','Std After'])
    return df

def eigen(data) : # find eigen vectors and eigenvalues
    cov=data.cov() # covariance matrice
    w,v=np.linalg.eig(cov)
    arr=sorted(w,reverse=True)  # sorting the eigen values in descending order
    eigvec=[]
    for i in range(len(w)):
        for j in range(len(w)) :
            if w[j]==arr[i] :
                eigvec.append(v[:,j])
    return eigvec,arr

def reduce(data,l) : # reduce dimension of data using PCA class
    pca=PCA(n_components=l)
    pca.fit(data)
    df=pd.DataFrame(pca.transform(data))
    return df
    
def scatter(x,y,xlabel,ylabel,title,c) : # scatter plot
    plt.subplots(figsize=(10,6))
    plt.scatter(x,y,color=c)
    plt.xlabel(xlabel,fontsize=14)
    plt.ylabel(ylabel,fontsize=14)
    plt.title(title,fontsize=18)
    plt.show()
 
def error(data,l,column) : # calculate reconstruction error
    pca=PCA(n_components=l)
    pca.fit(data)
    df=pd.DataFrame(pca.transform(data))
    print('Covariance matrice for l =',l)
    print(round(df.cov(),5))
    print()
    df2=pd.DataFrame(pca.inverse_transform(df))
    err=[]
    for i in range(768) :
        s=0
        for j in range(8) :
            x=(data[column[j]][i]-df2[j][i])**2
            s=s+x
        err.append(s**0.5)
    mean=np.mean(err)
    return mean
        
        