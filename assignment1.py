#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import numpy as np
import pandas as pd


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[4]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[5]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


# In[6]:


# Question 1
### Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked
random.seed(10)

bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()


return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/1.4: break


UsersPerItem = defaultdict(set)
ItemsPerUser = defaultdict(set)

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    UsersPerItem[b].add(u)
    ItemsPerUser[u].add(i)
    
threshold = 0.003
#predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        continue
    u,b = l.strip().split(',')
    Items = ItemsPerUser[u]
    #similarities = [0]
    #for b2 in Items:
        #if b2 != b : 
            #similarities.append(Jaccard(UsersPerItem[b],UsersPerItem[b2]))
            
    if b in return1:
        predictions.write(u + ',' + b + ",1\n")
    else:
        predictions.write(u + ',' + b + ",0\n")

predictions.close()


# In[ ]:





# In[7]:


# Question 2


# In[8]:


'''
userBiases = defaultdict(float)
itemBiases = defaultdict(float)
'''


# In[9]:


'''
userGamma = {}
itemGamma = {}
'''


# In[10]:


'''
K = 3
'''


# In[11]:


'''
def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)
'''


# In[12]:


'''
def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    global userGamma
    global itemGamma
    index = 0
    alpha = theta[index]
    index += 1
    userBiases = dict(zip(users, theta[index:index+nUsers]))
    index += nUsers
    itemBiases = dict(zip(items, theta[index:index+nItems]))
    index += nItems
    for u in users:
        userGamma[u] = theta[index:index+K]
        index += K
    for i in items:
        itemGamma[i] = theta[index:index+K]
        index += K
'''


# In[13]:


'''
def inner(x, y):
    return sum([a*b for a,b in zip(x,y)])
'''


# In[14]:


'''
def prediction(user, item):
    predict =  alpha + userBiases[user] + itemBiases[item] + inner(userGamma[user], itemGamma[item])
    return predict
'''


# In[15]:


'''
def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(d['userID'], d['bookID']) for index, d in data.iterrows()]
    cost = MSE(predictions, labels)
    print("MSE = " + str(cost))
    for u in users:
        cost += lamb*userBiases[u]**2
        for k in range(K):
            cost += lamb*userGamma[u][k]**2
    for i in items:
        cost += lamb*itemBiases[i]**2
        for k in range(K):
            cost += lamb*itemGamma[i][k]**2
    return cost
'''


# In[16]:


'''
def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(data)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    dUserGamma = {}
    dItemGamma = {}
    for u in users:
        dUserGamma[u] = [0.0 for k in range(K)]
    for i in items:
        dItemGamma[i] = [0.0 for k in range(K)]
    for idx, d in data.iterrows():
        u,i = d['userID'], d['bookID']
        pred = prediction(u, i)
        diff = pred - d['rating']
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[i] += 2/N*diff
        for k in range(K):
            dUserGamma[u][k] += 2/N*itemGamma[i][k]*diff
            dItemGamma[i][k] += 2/N*userGamma[u][k]*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
        for k in range(K):
            dUserGamma[u][k] += 2*lamb*userGamma[u][k]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
        for k in range(K):
            dItemGamma[i][k] += 2*lamb*itemGamma[i][k]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    for u in users:
        dtheta += dUserGamma[u]
    for i in items:
        dtheta += dItemGamma[i]
    return np.array(dtheta)
    
'''


# In[ ]:





# In[17]:


'''
lamb = 0.000004

f = gzip.open("train_Interactions.csv.gz", 'rt')
data = pd.read_csv(f)

ratingMean = data['rating'].mean()
alpha = ratingMean

labels = data['rating']

UsersPerItem = defaultdict(set)
ItemsPerUser = defaultdict(set)

users = list(set(data['userID']))
items = list(set(data['bookID']))
nUsers = len(users)
nItems = len(items)

for u in users:
    userGamma[u] = [random.random() * 0.1 - 0.05 for k in range(K)]
for i in items:
    itemGamma[i] = [random.random() * 0.1 - 0.05 for k in range(K)]

scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + # Initialize alpha
                                   [0.0]*(nUsers+nItems) + # Initialize beta
                                   [random.random() * 0.1 - 0.05 for k in range(K*(nUsers+nItems))], # Gamma
                             derivative, args = (labels, lamb), maxiter = 50)
'''


# In[18]:


'''
allRatings = []
userRatings = defaultdict(list)

for user,book,r in readCSV("train_Interactions.csv.gz"):
    r = int(r)
    allRatings.append(r)
    userRatings[user].append(r)

predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
    #header
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    
    if i in itemBiases and u in userBiases:
        predictions.write(u + ',' + b + ',' +  str(prediction(u, i)) + '\n')
    else:
        predictions.write(u + ',' + b + ',' + str(0) + '\n')

predictions.close()
'''


# In[19]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

trainRatings = [r[2] for r in ratingsTrain]
globalAverage = sum(trainRatings) * 1.0 / len(trainRatings)


# In[20]:


betaU = {}
betaI = {}
for u in ratingsPerUser:
    betaU[u] = 0

for b in ratingsPerItem:
    betaI[b] = 0
alpha = globalAverage # Could initialize anywhere, this is a guess
def iterate(lamb):
    newAlpha = 0
    for u,b,r in ratingsTrain:
        newAlpha += r - (betaU[u] + betaI[b])
    alpha = newAlpha / len(ratingsTrain)
    for u in ratingsPerUser:
        newBetaU = 0
        for b,r in ratingsPerUser[u]:
            newBetaU += r - (alpha + betaI[b])
        betaU[u] = newBetaU / (lamb + len(ratingsPerUser[u]))
    for b in ratingsPerItem:
        newBetaI = 0
        for u,r in ratingsPerItem[b]:
            newBetaI += r - (alpha + betaU[u])
        betaI[b] = newBetaI / (lamb + len(ratingsPerItem[b]))
    mse = 0
    for u,b,r in ratingsTrain:
        prediction = alpha + betaU[u] + betaI[b]
        mse += (r - prediction)**2
    regularizer = 0
    for u in betaU:
        regularizer += betaU[u]**2
    for b in betaI:
        regularizer += betaI[b]**2
    mse /= len(ratingsTrain)
    return mse, mse + lamb*regularizer
mse,objective = iterate(1)
newMSE,newObjective = iterate(1)
iterations = 1
while iterations < 10 or objective - newObjective > 0.0001:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate(5)
    iterations += 1
    print("Objective after " + str(iterations) + " iterations = " + str(newObjective))
    print("MSE after " + str(iterations) + " iterations = " + str(newMSE))
    
    
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if b in betaI:
        bi = betaI[b]
    _ = predictions.write(u + ',' + b + ',' + str(alpha + bu + bi) + '\n')

predictions.close()

