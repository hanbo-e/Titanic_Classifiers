"""
Machine Learning Workflow

1. Define Goal - Predict if titanic passenger will survive
2. Get Data - got it!
3. Train-Test-Split - done!
4. Explore Data
5. Feature Engineering
6. Train Model(s)
7. Optimize Hyperparameters / Cross Validation (Jump to Feature Engineering)
8. Calculate Test Score
9. Deploy and Monitor
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency

my_file = "data/train.csv"
df = pd.read_csv(my_file)
df.shape
df.head()
df.columns
df.dtypes
df.info()  # use output for finding NaNs

# TT-split
X = df.drop("Survived", axis=1)
y = df["Survived"]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# set passngerid as index
Xtrain.set_index("PassengerId", inplace=True)

# set categorical values as dtype category
num_cols = ["Age", "SibSp", "Parch", "Fare"]
for col in Xtrain.columns:
    if col not in num_cols:
        Xtrain[col] = Xtrain[col].astype("category")

# EDA
descriptives_num = Xtrain.describe(exclude=["category"])
descriptives_num

descriptives_cat = Xtrain.describe(include=["category"])
descriptives_cat

cat_cols = descriptives_cat.columns
cat_cols

# make contingency table for 'Survived' vs every category where unique < 10:
# Survived*Pclass

Pclass_crosstab = pd.crosstab(Xtrain["Survived"], Xtrain["Pclass"], margins=True)
Pclass_crosstab

Pclass_crosstab_norm = pd.crosstab(
    Xtrain["Survived"], Xtrain["Pclass"], margins=True, normalize=True
)
Pclass_crosstab_norm

# Survived*Sex
Sex_crosstab = pd.crosstab(Xtrain["Survived"], Xtrain["Sex"], margins=True)
Sex_crosstab

Sex_crosstab_norm = pd.crosstab(
    Xtrain["Survived"], Xtrain["Sex"], margins=True, normalize=True
)
Sex_crosstab_norm

# Survived*Embarked (embarked has null values!)
Embarked_crosstab = pd.crosstab(Xtrain["Survived"], Xtrain["Embarked"], margins=True)
Embarked_crosstab

Embarked_crosstab_norm = pd.crosstab(
    Xtrain["Survived"], Xtrain["Embarked"], margins=True, normalize=True
)
Embarked_crosstab_norm

# contingency tables without margins
Pclass_crosstab = pd.crosstab(Xtrain["Survived"], Xtrain["Pclass"])
Pclass_crosstab

# Survived*Sex
Sex_crosstab = pd.crosstab(Xtrain["Survived"], Xtrain["Sex"])
Sex_crosstab

# Survived*Embarked (embarked has null values!)
Embarked_crosstab = pd.crosstab(Xtrain["Survived"], Xtrain["Embarked"])
Embarked_crosstab

# chi-squared test embarked
stat, p, dof, expected = chi2_contingency(Embarked_crosstab)
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print("Dependent (reject H0) ")
else:
    print("Independent (keep H0) ")

# Embarked is dependant

# chi-squared test sex
stat, p, dof, expected = chi2_contingency(Sex_crosstab)
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print("Dependent (reject H0) ")
else:
    print("Independent (keep H0) ")
# Sex is dependant

# chi-squared test pclass
stat, p, dof, expected = chi2_contingency(Pclass_crosstab)
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print("Dependent (reject H0) ")
else:
    print("Independent (keep H0) ")
# Class is dependent

# chi for cabin
Cabin_crosstab = pd.crosstab(Xtrain["Survived"], Xtrain["Cabin"])
Cabin_crosstab
stat, p, dof, expected = chi2_contingency(Cabin_crosstab)
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print("Dependent (reject H0) ")
else:
    print("Independent (keep H0) ")
# Cabin is independent - but try recoding NaN to 'unknown' and adding that as a category

# chi for ticket
Ticket_crosstab = pd.crosstab(Xtrain["Survived"], Xtrain["Ticket"])
Ticket_crosstab
stat, p, dof, expected = chi2_contingency(Ticket_crosstab)
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print("Dependent (reject H0) ")
else:
    print("Independent (keep H0) ")
# Ticket is dependent - recode NaN to 'unknown, ticket probably correlates with class

# Do chi-squared tests for feature selection using skleans SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

y = df["Survived"]
X = df.select_dtypes(include=["category"])
X = X.drop(["Survived"], axis=1)
# replace NaNs with 'unknown'
# X.fillna('unknown', inplace=True)

chi2_selector = SelectKBest(chi2, k=3)
X_kbest = chi2_selector.fit_transform(X, y)
# Value Error, this function does not work with strings

# make a visualization of the cross tabs
import seaborn as sns

sns.set_theme(style="whitegrid")
sns.catplot(data=Xtrain, x="Survived", hue="Embarked", kind="count")
sns.catplot(
    data=Xtrain, x="Sex", y="Survived", hue="Embarked", kind="point"
)  # needs numeric

# trying statsmodels mosaic plot
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt

mosaic(Xtrain, ["Survived", "Pclass", "Sex"])
mosaic(Xtrain, ["Survived", "Pclass"])

#Create bins for the int/float data and convert to categorical
Xtrain['Age'] = pd.cut(Xtrain['Age'], 9, labels=["Child", "Teen", "Twenties", "Thirties", "Fourties", "Fifties", "Sixties",\
                                                 "Seventies", "Eighties"])
Xtrain['SibSp'] = pd.cut(Xtrain['SibSp'], 3, labels=["Low", "Medium", "High"])
Xtrain['Parch'] = pd.cut(Xtrain['Parch'], 3, labels=["Low", "Medium", "High"])
Xtrain['Fare'] = pd.cut(Xtrain['Fare'], 5, labels=["Very Low", "Low", "Medium", "High", "Very High"])

#drop name
Xtrain = Xtrain.drop(['Name'], axis = 1)

#Make chi-square heatmap (code taken from "Analytics Vidhya" Medium article))
#NEED TO ADD SURVIVED!

col_names = Xtrain.columns
#ChiSqMatrix = pd.DataFrame(Xtrain, columns=col_names, index=col_names)
ChiSqMatrix = pd.DataFrame(columns=col_names, index=col_names, dtype=np.dtype("float"))
#loop through values and get chi-square scores
outcount = 0
incount = 0
for icol in col_names:
    for jcol in col_names:
        myCrossTab=pd.crosstab(Xtrain[icol], Xtrain[jcol])
        stat, p, dof, expected = chi2_contingency(myCrossTab)
        ChiSqMatrix.iloc[outcount, incount]=round(p, 5)
        #check that expected freq at least 5 for 80% of cells
        countExpected=expected[expected<5].size
        percentExpected=((expected.size-countExpected)/expected.size)*100
        if percentExpected<20:
            ChiSqMatrix.iloc[outcount, incount]=2
        if icol == jcol:
            ChiSqMatrix.iloc[outcount, incount]=0.00
        incount += 1
    outcount += 1
    incount = 0

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 11))
mask = np.triu(np.ones_like(ChiSqMatrix, dtype=np.bool))
sns.heatmap(ChiSqMatrix, mask=mask, cmap="Blues", annot=True)
plt.show()
"""
TODOs
Reduce 'Cabin' data to 'Deck' data
Create Facet grid of barplots of each variable to survival
Convert all data to int format for SelectKBest function
"""
