# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 22:29:06 2021

@author: Titiksha
Roll No.: B20138
Mobile no.: 9811162646
"""
import pandas as pd
import functions as f
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data_ini=pd.read_csv("pima-indians-diabetes.csv",sep=',')

# Question 1
column=list(data_ini.columns)
data=f.replace_out(data_ini,column)  # replace outlier values with median

# part a
data_nor=pd.DataFrame()   
for i in range(8):  # Normalise indivial attributes
    data_nor[column[i]]=f.normal(data[column[i]]) 
print('Question 1: part a')
print(f.BA_nor(data,data_nor,column))
print()

# part b
# standarise the data using StandardScaler class
scaler=StandardScaler()
scaler.fit(data)
data_std=pd.DataFrame(scaler.transform(data),columns=column[:8])
print('part b:')
print(f.BA_std(data,data_std,column))
print()

# Question 3 --> part a
eigvec,eigval=f.eigen(data_std)  # find eigen values for the data
df_reduced=f.reduce(data_std,2)  # reduce the dimension to l=2
# compare variance and eigen values
print('Question 3: part a')
print('Variance along direction 1 =',round(np.var(df_reduced[0]),5))
print('Eigen value for direction 1 =',round(eigval[0],5))
print()
print('Variance along direction 2 =',round(np.var(df_reduced[1]),5))
print('Eigen value for direction 2 =',round(eigval[1],5))
print()
# scatter plot of reduced data
f.scatter(df_reduced[0],df_reduced[1],'x-axis','y-axis','Plot of data after dimensionality reduction','orange')

# part b --> scatte plot of eigen values
index=[i for i in range(1,len(eigval)+1)]
f.scatter(index,eigval,'','Eigenvalues','Plot of eigenvalues in descending order','mediumvioletred')

# part c 
print('part c:')
error=[]
for i in range(1,9) :
    error.append(f.error(data_std,i,column))
plt.subplots(figsize=(10,6)) # line plot of reconstruction error
plt.plot(index,error,color='mediumvioletred')
plt.xlabel('No. of reduced dimension',fontsize=14)
plt.ylabel('Reconstruction errror',fontsize=14)
plt.title('Line plot to demonstrate reconstruction error vs. no. of dimensions',fontsize=18)
plt.show()

# part d
print('part d: Covariance matrice for the original data')
print(data_std.cov())






