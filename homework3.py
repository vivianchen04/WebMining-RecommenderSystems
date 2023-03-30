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


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[5]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[6]:


answers = {}


# In[7]:


# Some data structures that will be useful


# In[8]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[9]:


len(allRatings)


# In[10]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[11]:


##################################################
# Rating prediction (CSE258 only)                #
##################################################


# In[12]:


import tensorflow as tf


# In[13]:


userIDs = {}
itemIDs = {}

for d in allRatings:
    u = d[0]
    i = d[1]
    r = d[2]
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)


# In[14]:


### Question 9


# In[15]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

mu = sum([r for _,_,r in ratingsTrain]) / len(ratingsTrain)

class LatentFactorModelBiasOnly(tf.keras.Model):
    def __init__(self, mu, lamb):
        super(LatentFactorModelBiasOnly, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu)
        # Initialize to small random values
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.lamb = lamb

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        p = self.alpha + self.betaU[u] + self.betaI[i]
        return p
    
    def user_beta(self, u):
        return self.betaU[u]
    
    # Regularizer
    def reg(self):
        return self.lamb * (tf.reduce_sum(self.betaU**2) +\
                            tf.reduce_sum(self.betaI**2))
    
    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        pred = self.alpha + beta_u + beta_i
        return pred
    
    # Loss
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)
    
modelBiasOnly = LatentFactorModelBiasOnly(mu, 1) # lamb = 1
labels = [r for _,_,r in ratingsValid]

biasOnlyPredictions =\
    [modelBiasOnly.predict(userIDs[u],itemIDs[i]).numpy() for u,i,_ in ratingsValid]
validMSE = MSE(biasOnlyPredictions, labels)
validMSE


# In[16]:


biasforUsers = \
    [[u, modelBiasOnly.user_beta(userIDs[u])] for u,i,_ in ratingsValid]


# In[17]:


min(biasforUsers[:50], key = lambda x: x[1])


# In[18]:


answers['Q9'] = validMSE


# In[19]:


assertFloat(answers['Q9'])


# In[20]:


### Question 10


# In[21]:


max(biasforUsers[:50], key = lambda x: x[1])
min(biasforUsers[:50], key = lambda x: x[1])
maxUser = 'u08390130'
minUser = 'u42941492'
maxBeta = 0.0019333808
minBeta = -0.0023056408


# In[22]:


answers['Q10'] = [maxUser, minUser, maxBeta, minBeta]


# In[23]:


assert [type(x) for x in answers['Q10']] == [str, str, float, float]


# In[24]:


### Question 11


# In[25]:


for lamb in (0.001, 0.01, 0.1, 10, 100):
    modelBiasOnly = LatentFactorModelBiasOnly(mu,lamb) 
    biasOnlyPredictions =\
        [modelBiasOnly.predict(userIDs[u],itemIDs[i]).numpy() for u,i,_ in ratingsValid]
    validMSE = MSE(biasOnlyPredictions, labels)
    print('MSE: {} lamb: {} '.format(validMSE, lamb))
    


# In[26]:


lamb, validMSE = 10, 1.6801640138536982


# In[27]:


answers['Q11'] = (lamb, validMSE)


# In[28]:


assertFloat(answers['Q11'][0])
assertFloat(answers['Q11'][1])
answers


# In[29]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
predictions.close()


# In[30]:


##################################################
# Read prediction                                #
##################################################


# In[31]:


# Copied from baseline code
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
    if count > totalRead/2: break


# In[32]:


### Question 1


# In[33]:


UserItem = defaultdict(set)
total = set()
random.seed(0)

for u, b, _ in allRatings:
    UserItem[u].add(b)
    total.add(b)

neg = []
for user, _, _ in ratingsValid:
    neg.append(random.sample(total - UserItem[user], 1))

valid = 0
for i in range(len(ratingsValid)):
    if ratingsValid[i][1] in return1:
        valid += 1
    if neg[i][0] not in return1:
        valid += 1
acc1 = valid / (2*len(ratingsValid))


# In[34]:


answers['Q1'] = acc1
acc1


# In[35]:


assertFloat(answers['Q1'])


# In[36]:


### Question 2


# In[37]:


return2 = set()
summ = 0
for new_count, i in mostPopular:
    if new_count < 50: 
        break
    summ += new_count
    return2.add(i)

valid = 0
for i in range(len(ratingsValid)):
    if neg[i][0] not in return2:
        valid += 1
    if ratingsValid[i][1] in return2:
        valid += 1


# In[38]:


acc2 = valid / (2 * len(ratingsValid))
threshold = summ/totalRead


# In[39]:


answers['Q2'] = [threshold, acc2]


# In[40]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[41]:


### Question 3/4


# In[42]:


UserItem = defaultdict(set)
total = set()
random.seed(0)

for u, b, _ in allRatings:
    UserItem[u].add(b)
    total.add(b)

i = 0
negative_all = ratingsValid
for user, book, _ in ratingsValid:
    temp = random.sample(total - UserItem[user], 1)[0]
    negative_all.append((user, temp, -1))
    i += 1
    if i == 10000:
        break
    


# In[43]:





# In[ ]:





# In[44]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


# In[46]:


UPerI = defaultdict(set)
IPerU = defaultdict(set)
for u,i,_ in ratingsTrain:
    UPerI[i].add(u)
    IPerU[u].add(i)

TH = np.arange(0,0.01,0.001)

for threshold in TH:
    valid = 0
    for user,b,rating in negative_all:
        Items = IPerU[user]
        s = [0]
        for b2 in Items:
            if b2 != b : 
                s.append(Jaccard(UPerI[b],UPerI[b2]))
        if max(s) > threshold:
            valid += (rating != -1)
        else:
            valid += (rating == -1)
    acc3 = valid/len(negative_all)
    print("threshold: {} acc: {}".format(threshold, acc3))
    


# In[ ]:


acc3 = 0.69585


# In[47]:


# Q4
def feature():
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/2: break
    return return1

threshold = 0.003
valid = 0
for user,b,rating in negative_all:
    Items = IPerU[user]
    s = [0]
    for b2 in Items:
        if b2 != b : 
            s.append(Jaccard(UPerI[b],UPerI[b2]))
    if max(s) > threshold and b in feature():
        valid += (rating != -1)
    else:
        valid += (rating == -1)
acc4 = valid/len(negative_all)
acc4


# In[48]:


answers['Q3'] = acc3
answers['Q4'] = acc4


# In[49]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[50]:


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[51]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"
answers


# In[52]:


assert type(answers['Q5']) == str


# In[ ]:





# In[ ]:


##################################################
# Category prediction (CSE158 only)              #
##################################################


# In[ ]:


### Question 6


# In[ ]:


data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)


# In[ ]:


data[0]


# In[ ]:





# In[ ]:


answers['Q6'] = counts[:10]


# In[ ]:


assert [type(x[0]) for x in answers['Q6']] == [int]*10
assert [type(x[1]) for x in answers['Q6']] == [str]*10


# In[ ]:


### Question 7


# In[ ]:





# In[ ]:


Xtrain = X[:9*len(X)//10]
ytrain = y[:9*len(y)//10]
Xvalid = X[9*len(X)//10:]
yvalid = y[9*len(y)//10:]


# In[ ]:





# In[ ]:


answers['Q7'] = acc7


# In[ ]:


assertFloat(answers['Q7'])


# In[ ]:


### Question 8


# In[ ]:





# In[ ]:


answers['Q8'] = acc8


# In[ ]:


assertFloat(answers['Q8'])


# In[ ]:


# Run on test set


# In[ ]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[ ]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




