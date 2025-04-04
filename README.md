# Customer Churn Prediction

## Overview
This project aims to predict customer churn using machine learning models. Various classification techniques are implemented, including Logistic Regression, Random Forest, SVM, and XGBoost.

## Dataset
The dataset used is **"Churn Modeling.csv"**, which contains customer data with features such as age, balance, tenure, and credit score.

## Project Workflow
### ## Importing necessary libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
... (truncated)
```

```python
df=pd.read_csv("Churn Modeling.csv")
# Reading dataset
... (truncated)
```

```python
df
# loading dataset
... (truncated)
```

### ## Data Understanding

```python
df.shape
# There are 10,000 rows and 14 columns
... (truncated)
```

```python
df.info()
# Checking the information of the data
... (truncated)
```

```python
df.isnull().sum()
# There are no null values in the dataset
... (truncated)
```

```python
df['Exited'].value_counts()
# As it can be seen this dataset's targer column is imbalanced there are 7963 zero's and 2037 one's
... (truncated)
```

```python
df.describe()
# The describe function will display all the decriptive statistics of the data including mean, std, min, max values.
... (truncated)
```

```python
df.nunique()
# The nunique() function is used to count distinct observations over requested axis. Return Series with number of distinct observations.
... (truncated)
```

```python
df.columns
# Displaying the column names of the dataset.
... (truncated)
```

### ## Data Visualization

```python
sns.pairplot(data=df)
# Displaying scatter plots
... (truncated)
```

```python
df.columns
# Displaying column names
... (truncated)
```

```python
df.head(5)
# Displaying the head of the dataset
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='Geography',y='EstimatedSalary',hue='Gender',color='teal',data=df);
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='Age',y='EstimatedSalary',data=df);
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='Age',y='Exited',color='teal',data=df);
... (truncated)
```

```python
plt.figure(figsize=(10, 8))
plt.xticks(rotation=90)
sns.barplot(x='Gender',y='EstimatedSalary',color='orange',data=df);
... (truncated)
```

```python
plt.figure(figsize=(10, 8))
plt.xticks(rotation=90)
sns.barplot(x='Gender',y='Balance',color='teal',data=df);
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='Age',y='EstimatedSalary',hue='Gender',data=df);
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='Age',y='NumOfProducts',data=df);
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='NumOfProducts',y='EstimatedSalary',data=df);
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='Age',y='Tenure',hue='Gender',data=df);
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='IsActiveMember',y='Geography',hue='Gender',data=df);
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='IsActiveMember',y='Exited',hue='Gender',data=df);
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='IsActiveMember',y='EstimatedSalary',hue='Gender',data=df);
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='HasCrCard',y='Exited',hue='Gender',data=df); 
... (truncated)
```

```python
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.barplot(x='HasCrCard',y='Geography',hue='Gender',data=df);
... (truncated)
```

### ## Label Encoding

```python
cat_cols=['Geography','Gender']
le=LabelEncoder()
for i in cat_cols:
... (truncated)
```

```python
df.keys()
# displaying columns
... (truncated)
```

```python
df.drop(['RowNumber'],axis=1,inplace=True)
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
... (truncated)
```

### ## DistributionPlot

```python
rows=2
cols=5
fig, ax=plt.subplots(nrows=rows,ncols=cols,figsize=(16,4))
... (truncated)
```

```python
X=df.drop(labels=['Exited'],axis=1)
Y=df['Exited']
X.head()
... (truncated)
```

```python
Y.head()
# This is the target column
... (truncated)
```

```python
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=40)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
# Splitting the data set into training and testing data
... (truncated)
```

### ### Logistic Regression

```python
#fit the model on train data 
log_reg = LogisticRegression().fit(X_train, Y_train)

... (truncated)
```

### ### Naive Bayes Classifier

```python
#fit the model on train data 
NB=GaussianNB()
NB.fit(X_train,Y_train)
... (truncated)
```

### ### Decision Tree Classifier

```python
#fit the model on train data 
DT = DecisionTreeClassifier().fit(X,Y)

... (truncated)
```

### ### Random Forest Classifier

```python
#fit the model on train data 
RF=RandomForestClassifier().fit(X_train,Y_train)
#predict on train 
... (truncated)
```

### ### K-Nearest Neighbours

```python
#fit the model on train data 
KNN = KNeighborsClassifier().fit(X_train,Y_train)
#predict on train 
... (truncated)
```

### ### Support Vector Machine

```python
#fit the model on train data 
SVM = SVC(kernel='linear')
SVM.fit(X_train, Y_train)
... (truncated)
```

### ### XG-Boost Classifier

```python
xgbr =xgb.XGBClassifier().fit(X_train, Y_train)

#predict on train 
... (truncated)
```

```python
# Random Forest Classifier and Xg-boost Regressor model performed well compared to other models
... (truncated)
```

### # Hyper Parameter Tuning

```python
#fit the model on train data 
RFT=RandomForestClassifier(n_estimators=500,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',random_state=235,verbose=2,max_samples=50).fit(X_train,Y_train)
#predict on train 
... (truncated)
```

### ### A. RandomSearchCv

```python
from sklearn.model_selection import RandomizedSearchCV
... (truncated)
```

```python
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 5000, num = 10)]
# Number of features to consider at every split
... (truncated)
```

```python
RFT1=RandomForestClassifier()
rf_randomcv=RandomizedSearchCV(estimator=RFT1,param_distributions=random_grid,n_iter=100,cv=5,verbose=2,
                               random_state=100,n_jobs=-1)
... (truncated)
```

```python
rf_randomcv.best_params_
# best parameters
... (truncated)
```

```python
rf_randomcv
# displaying all parameters
... (truncated)
```

```python
rf_randomcv.best_estimator_
# Displaying best parameters from all parameters mentioned above
... (truncated)
```

```python
best_random_grid=rf_randomcv.best_estimator_
# saving all parameters in best_random_grid
... (truncated)
```

```python
Y_pred=best_random_grid.predict(X_test)

print(confusion_matrix(Y_test,Y_pred))
... (truncated)
```

### ### B. GridSearchCv

```python
from sklearn.model_selection import GridSearchCV
# Using grid search cv model
... (truncated)
```

```python
rf_randomcv.best_params_
# Visualizing the best parameters of random cv
... (truncated)
```

```python
param_grid = {
    'criterion': [rf_randomcv.best_params_['criterion']],
    'max_depth': [rf_randomcv.best_params_['max_depth']],
... (truncated)
```

```python
rf=RandomForestClassifier()
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=10,n_jobs=-1,verbose=2)
grid_search.fit(X_train,Y_train)
... (truncated)
```

```python
grid_search.best_estimator_
# best parameters of grid search
... (truncated)
```

```python
best_grid=grid_search.best_estimator_
# saving the parameters in best_grid
... (truncated)
```

```python
best_grid
# Displaying best_grid
... (truncated)
```

```python
y_pred=best_grid.predict(X_test)
print(confusion_matrix(Y_test,y_pred))
print("Accuracy Score {}".format(accuracy_score(Y_test,y_pred)))
... (truncated)
```

### #### The decision tree model showed us overfitting problem.
#### Hence the randomised search cv on random forest classifier gave us better accuracy which is 87 percent and wrong predictions made by the model are 243/2000 and grid search cv gave us 87 percent accuracy and wrong predictions are 246/2000.

## Installation & Requirements
Ensure the following Python libraries are installed:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

## How to Run
1. Load the dataset.
2. Preprocess the data (handle missing values, encode categorical features, scale numerical features).
3. Train different machine learning models.
4. Evaluate model performance using accuracy, ROC-AUC score, and confusion matrix.

## Results & Evaluation
- The best performing model is selected based on accuracy and ROC-AUC score.
- Performance is visualized using confusion matrices and ROC curves.

## Challenges Faced & Solutions
- **Data Imbalance**: Addressed using SMOTE for oversampling.
- **Feature Selection**: Used correlation heatmaps and feature importance methods.
- **Hyperparameter Tuning**: Implemented GridSearchCV for optimizing models.




## Project Discussion

**Describe a recent project.**
This project focuses on predicting customer churn using machine learning techniques. It involves data preprocessing, feature selection, model training, and evaluation.

**Why did you choose this project?**
Customer retention is crucial for businesses, and predicting churn allows companies to take proactive measures. This project provides insights into customer behavior and identifies at-risk customers.

**What were the biggest challenges, and how did you overcome them?**
1. **Handling imbalanced data** – Used oversampling techniques like SMOTE.
2. **Feature Engineering** – Selected the most important features using correlation analysis.
3. **Choosing the best model** – Experimented with multiple classifiers and selected the best based on performance metrics.

**Showcase the project via screen share.**
Run the notebook in Jupyter or Google Colab, visualize the data insights, and demonstrate model training results.
