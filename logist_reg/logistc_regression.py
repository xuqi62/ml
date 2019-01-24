#!/usr/bin/env python3
# coding: utf-8

# ## Logistic Regression
# 使用Logistic回归解决二分类问题，数据来源于kaggle手写数字识别数据集。
# 
# 数字0为类别0;数字1-9为类别1.
# 
# 

# In[25]:


# %load lr.py
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LogisticRegression:
    def __init__(self):
        print("init 11111")
        self.__learning_rate = 0.000001
        self.__lamda = 0.01
        
    def __sigmoid(self, x):    
        return 1.0/(1+ np.exp(-x))      
    
    def gradDesc(self, xArr, yArr):
        n,m = xArr.shape      
        
        print("xArr.shape:", xArr.shape, yArr.shape)
        self.__W = np.zeros((1, n), dtype=np.float64)
        self.__b = 0.0
        print(self.__W.shape)
        step = 1000
        alpha = self.__learning_rate
        
        for i in range(step):
            y_hat = self.__sigmoid(self.__W.dot(xArr) + self.__b)
            loss = y_hat - yArr

            self.__W = self.__W - alpha*loss.dot(xArr.T)
            self.__b = self.__b - alpha*loss
            if(i%100 == 0):
                ones_array = np.ones((1, m))           
                loss_arr = -yArr*np.log(y_hat)-(ones_array-yArr)*np.log(ones_array - y_hat)
                loss = np.sum(loss_arr, axis=1)
                print("loss sum:", loss)
        return self.__W
    
    def predict(self, xArr, yArr):
        n, m = xArr.shape

        y_hat = self.__sigmoid(self.__W.dot(xArr) + self.__b)
        print("y_hat shape:", y_hat.shape)
        
        right = 0
        for i in range(m):
            if((yArr[0][i]==1) and (y_hat[0][i]>0.5)):
                right += 1
            elif ((yArr[0][i]==0) and (y_hat[0][i]<=0.5)):
                right += 1
        print ("rate:", right / m)
        return

if __name__ == "__main__":
    lr = LogisticRegression();
    
    raw_data = pd.read_csv("train.csv", header=0)
    values = raw_data.values
    
    m, n = values.shape
    print ("data shape:", values.shape)
    
    dataTrain = values[0::, 1::]

    label = values[0::, 0]
    labelTrain = []
    for i in range(m):
        if(label[i] > 0):
            labelTrain.append(1)
        else:
            labelTrain.append(0)
    
    xArr = np.array(dataTrain, dtype=np.float64).T / 255.0
    n, m = xArr.shape
    yArr = np.array(labelTrain, dtype=np.float64).reshape(1, m)
    
    print("data shape:", dataTrain.shape)
    
    W1 = lr.gradDesc(xArr, yArr)
    
    lr.predict(xArr, yArr)

