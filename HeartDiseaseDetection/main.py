import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import plotly
import plotly.express as px
import plotly.offline as pyo

from plotly.offline import init_notebook_mode,plot,iplot

import cufflinks as cf # binds pandas and plotly together

pyo.init_notebook_mode(connected=True)
cf.go_offline()

dataFrame = pd.read_csv("./data/heart.csv")

dataFrame.hist(figsize=(14,14))

### data preprocessing

X, y = dataFrame.loc[:,:'thal'],dataFrame['target']

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




#Standard scaling of the neighbour items for the standard normal distribution,
# data members will be scaled to a particular range so that KNN can calculated neighbouring items
# distance more precisely
std = StandardScaler().fit(X)
X_std = std.transform(X)

#using train_test_split to split 70% of data for training and 30% of the data for testing.
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=10,test_size=0.3,shuffle=True)


#Decision tree classifier model testing

from sklearn.tree import DecisionTreeClassifier


#using train_test_split to split 70% of data for training and 30% of the data for testing.
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=10,test_size=0.3,shuffle=True)

model = DecisionTreeClassifier()

model.fit(X_train,y_train) # training the 70% of the data


prediction =  model.predict(X_test) # predicting the value for the rest remaining 30% of data,returns an array

# Calculating the accuracy of the trained data

from sklearn.metrics import accuracy_score

accuracy_model= accuracy_score(y_test,prediction) # it return the value of 0.747252.. i.e prediction is around 74.72% correct

## creating a function for feature importance

def plot_feature(model):
    plt.figure(figsize=(8,6))
    no_of_features=13
    plt.barh(range(no_of_features),model.feature_importances_,align='center')
    #plt.show()

plot_feature(model) #ploting the most to least important feature determining the prediction of the data

#Check on some custom data

customData = np.array([[57,0,0,140,241,1,123,1,0.2,1,1,0,3]])

customPredict = model.predict(customData) #predicting a special case , which returns 1 for true and 0 for false.

#Using KNN algorithm to train the dataset

from sklearn.neighbors import KNeighborsClassifier


#Taking the standScaler X for the KNN algorithm for the normal distribution of the elements
X_train_std,X_test_std,y_train,y_test = train_test_split(X,y,random_state=10,test_size=0.3,shuffle=True)

knn = KNeighborsClassifier()
knn.fit(X_train_std,y_train)

predictionKNN= knn.predict(X_test_std)
accuracy_knn_model = accuracy_score(y_test,predictionKNN) ## calculating the accuracy of the trained dataset
customDataKNN = np.array([[57,0,0,140,241,1,123,1,0.2,1,1,0,3]])
customDataKNNStandard = std.transform(customDataKNN) #tranforming the custom data to normal distribution form


## Training the dataset for KNN for neighbouring values 1-25 and ploting it later to
# check the best value for neighbouring elements.
KnnRange = range(1,26)
bestNeighbouringValue =[]
for k in KnnRange:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    predictionKNN = knn.predict(X_test_std)
    bestNeighbouringValue.append(accuracy_score(y_test,predictionKNN))


plt.plot(KnnRange,bestNeighbouringValue)

algorithms = ['DescisionTree','KNN']

scores=[accuracy_model,accuracy_knn_model]
plt.bar(algorithms,scores)

plt.show()














