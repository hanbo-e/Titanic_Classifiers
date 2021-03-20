###compare function

#imports
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def compare_models(Xtrain, ytrain, Xtest, ytest):
    '''
    Parameters
    ----------
    Xtrain : DataFrame
        Training data.
    ytrain : Series
        y Training.
    Xtest : DataFrame
        Testing data.
    ytest : Series
        y testing.
    Returns
    -------
    Dictionary with scores for LR, DT, and RF.
    '''
    output = {'LR train score':[], 'LR test score': [], 'DT train score':[], 'DT test score': [], 'RF train score':[], 'RF test score': []}
    
    #create LR model and add scores to output
    LR = LogisticRegression()
    LR.fit(Xtrain, ytrain)
    output['LR train score'] = LR.score(Xtrain, ytrain)
    output['LR test score'] = LR.score(Xtest, ytest)
    #output['LR proba'] = LR.predict_proba([[3], [2], [1]])
    #output['LR predict'] = LR.predict([[3], [2], [1]])
    
    #create DT model and add scores to output
    DT = DecisionTreeClassifier()
    DT.fit(Xtrain, ytrain)
    output['DT train score'] = DT.score(Xtrain, ytrain)
    output['DT test score'] = DT.score(Xtest, ytest)
    #output['DT proba'] = DT.predict_proba([[3], [2], [1]])
    #output['DT predict'] = DT.predict([[3], [2], [1]])
    
    #create random forest and add scores to output
    RF = RandomForestClassifier()
    RF.fit(Xtrain, ytrain)
    output['RF train score'] = RF.score(Xtrain, ytrain)
    output['RF test score'] = RF.score(Xtest, ytest)
    #output['RF proba'] = RF.predict_proba([[3], [2], [1]])
    #output['RF predict'] = RF.predict([[3], [2], [1]])
    
    #return
    return output
#####main

#load data
df = pd.read_csv('train.csv')

#select X and y for LogisticRegression 'Pclass' feature only
X1 = df[['Pclass']]
y = df['Survived']

#train-test-split
X1train, X1test, ytrain, ytest = train_test_split(X1, y, test_size = 0.2, random_state = 42)

#call compare
my_out1 = compare_models(X1train, ytrain, X1test, ytest)

#keep adding features and compare
X2 = df[['Pclass', 'Sex']]
X2['Sex'].replace({'male':1, 'female': 0}, inplace = True)
X2train, X2test, ytrain, ytest = train_test_split(X2, y, test_size = 0.2, random_state = 42)

my_out2 = compare_models(X2train, ytrain, X2test, ytest)

#Three features
#check if missing values in age
my_null = df['Age'].isnull()
my_null.value_counts()
#replace with median
df['Age'].fillna(value = df['Age'].median(), inplace=True)

X3 = df[['Pclass', 'Sex', 'Age']]
X3['Sex'].replace({'male':1, 'female': 0}, inplace = True)
X3train, X3test, ytrain, ytest = train_test_split(X3, y, test_size = 0.2, random_state = 42)

y_out3 = compare_models(X3train, ytrain, X3test, ytest)

#print output
for key in my_out1:
    print(key, '\t1\t', my_out1[key])
print('\n')
for key in my_out2:
    print(key, '\t2\t', my_out2[key])
print('\n')
for key in y_out3:
     print(key, '\t3\t', y_out3[key])
print('\n')
