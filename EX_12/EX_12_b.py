import pandas as pd
from sklearn.datasets import load_wine
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt



df= pd.read_csv('covtype.csv')

X=np.array(df.drop(['Cover_Type'], axis=1))
Y=df['Cover_Type'].values

accs=[]
accstrain=[]
for i in range(0,10):
    Xtrain, Xtest, Ytrain, Ytest=train_test_split(X,Y,test_size=0.3)
    #model= DecisionTreeClassifier(max_depth=3)
    model=RandomForestClassifier(max_depth=3)
    model.fit(Xtrain,Ytrain)
    
    trainprediction=model.predict(Xtrain)
    testprediction=model.predict(Xtest)
    
    print('Train data accuracy %.2f'% accuracy_score(Ytrain,trainprediction))
    print('Test data accuracy %.2f'% accuracy_score(Ytest,testprediction))
    accs.append(accuracy_score(Ytest,testprediction))
    accstrain.append(accuracy_score(Ytrain,trainprediction))
    
print('Mean train data accuracy %.2f'% np.mean(accstrain))    
print('Mean test data accuracy %.2f'% np.mean(accs))
    
#plt.figure(figsize=[10,10])
#plot_tree(model,feature_names=df.drop(['target'], axis=1).columns)

#%% Print feature importances
plt.figure()
plt.barh(df.drop(['Cover_Type'], axis=1).columns,model.feature_importances_)
plt.tight_layout
plt.show()