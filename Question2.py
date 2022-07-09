# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 21:32:11 2021

@author: Titiksha
"""
# Question 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# generating the 2-dimensional synthetic data
# bi-variate gaussian distribution
mean=[0,0]
cov=[[13,-3],[-3,5]]
x,y=np.random.multivariate_normal(mean,cov,1000).T  

# part a --> scatter plot of 2-D data
plt.subplots(figsize=(10,6))
plt.scatter(x,y,color='teal',marker="x")
plt.xlabel('x-axis',fontsize=14)
plt.ylabel('y-yaxis',fontsize=14)
plt.title('Scatter plot of 2D synthetic data',fontsize=18)
plt.show()

# part b --> get eigen vectors and eigen values
w,v=np.linalg.eig(cov)
e1=v[:,0]
e2=v[:,1]

# plot eigen vectors on scatter plot
plt.subplots(figsize=(10,6))
plt.scatter(x,y,color='teal',label='Original data',marker="x")
plt.xlabel('x-axis',fontsize=14)
plt.ylabel('y-yaxis',fontsize=14)
plt.quiver(0,0,e1[0],e1[1],color='red',scale=3,label='Eigenvectors')
plt.quiver(0,0,e2[0],e2[1],color='red',scale=5)
plt.title('Scatter plot of 2D synthetic data and eigen directions',fontsize=18)
plt.legend()
plt.show()

def reduce(x,y,e) :  # reduce dimension
    new=[]
    for i in range(1000) :
        a=(e[0]*x[i])+(e[1]*y[i])
        new.append(a)
    return new
    
# part c --> projecting on eigen vectors
n1=reduce(x,y,e1)
x1=[(n1[i]*e1[0]) for i in range(1000)]
y1=[(n1[i]*e1[1]) for i in range(1000)]

plt.subplots(figsize=(10,6))
plt.scatter(x,y,color='teal',label='Original data',marker="x")
plt.quiver(0,0,e2[0],e2[1],color='red',label='Eigenvectors',scale=5)
plt.scatter(x1,y1,color='orange',label='Data projected on first eigen direction',marker="x")
plt.xlabel('x-axis',fontsize=14)
plt.ylabel('y-yaxis',fontsize=14)
plt.title('Projected values on the first eigen direction',fontsize=18)
plt.legend()
plt.show()

n2=reduce(x,y,e2)
x2=[(n2[i]*e2[0]) for i in range(1000)]
y2=[(n2[i]*e2[1]) for i in range(1000)]

plt.subplots(figsize=(10,6))
plt.scatter(x,y,color='teal',label='Original data',marker="x")
plt.quiver(0,0,e1[0],e1[1],color='red',label='Eigenvectors',scale=3)
plt.scatter(x2,y2,color='orange',label='Data projected on second eigen direction',marker="x")
plt.xlabel('x-axis',fontsize=14)
plt.ylabel('y-yaxis',fontsize=14)
plt.title('Projected values on the second eigen direction',fontsize=18)
plt.legend()
plt.show()

# part d --> reconstruction error
data=pd.DataFrame(list(zip(x,y)))
pca=PCA(n_components=2)
pca.fit(data)
df=pd.DataFrame(pca.transform(data)) # reducing dimension using PCA
df2=pd.DataFrame(pca.inverse_transform(df)) # reconstruct using PCA
err=[]
for i in range(1000) :
        s=((data[0][i]-df2[0][i])**2)+((data[1][i]-df2[1][i])**2)
        err.append(s**0.5)
print()
print('Reconstruction error =',(np.mean(err)))

