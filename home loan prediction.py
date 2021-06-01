import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# copy the original data
train_og = train.copy()
test_og = test.copy()

# look at the structure fo the datasets
# check the features present in the data

train.columns
test.columns

# Determine the shape of the dataset
print(f'Training data shape: {train.shape}')
train.head()

print(f'Test data shape: {test.shape}')
test.head()

# univariate analysis: examine each variable individually. for categoric features, frequency tables and bar plots will be used
# for numerical, probability density plots used to look at distribution of the variable

# the target variable is Loan status
# how many is in the variable
train['Loan_Status'].count()

# counting the different variables
train['Loan_Status'].value_counts()

# Normalize can be set to True to print proportions (percentage) instead of number
train['Loan_Status'].value_counts(normalize=True)*100

# plot a bar graph
train['Loan_Status'].value_counts(normalize=True).plot.bar(title='Loan_Status')
plt.show()

# Percentage of Men to Women
train['Gender'].value_counts(normalize=True)*100

# plot a bar graph
train['Gender'].value_counts(normalize=True).plot.bar(title='Gender')
plt.show()

# Percentage of Married to unmarried
train['Married'].value_counts(normalize=True)*100

# plot a bar graph
train['Married'].value_counts(normalize=True).plot.bar(title='Married')
plt.show()

# Percentage of Self employed to employed
train['Self_Employed'].value_counts(normalize=True)*100

# plot a bar graph
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')
plt.show()

# Percentage of Repaid debts to unrepaid debts
train['Credit_History'].value_counts(normalize=True)*100

# plot a bar graph
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
plt.show()

# Percentage of Dependents by number
train['Dependents'].value_counts(normalize=True)*100

# plot a bar graph
train['Dependents'].value_counts(normalize=True).plot.bar(title='Dependents')
plt.show()

# Percentage of Graduates to non graduates
train['Education'].value_counts(normalize=True)*100

# plot a bar graph
train['Education'].value_counts(normalize=True).plot.bar(title='Education', figsize=(10,10))
plt.show()

# Percentage of Property areas by type
train['Property_Area'].value_counts(normalize=True)*100

# plot a bar graph
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property Area', figsize=(10,10))
plt.show()

# These features have numerical values
# Applicant Income distribution
plt.figure(1)
plt.subplot(121)
sns.histplot(train['ApplicantIncome'])

# Drawing the second graph in the same picture, next to the other picture
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()

# showing the applicant income data separated by education
train.boxplot(column='ApplicantIncome', by='Education')
plt.suptitle('')
plt.show()

# Co applicants distribution
plt.figure(2)
plt.subplot(221)
sns.histplot(train['CoapplicantIncome'])

plt.subplot(222)
train['CoapplicantIncome'].plot.box(figsize=(16,10))
plt.show()

# sort out applicant income by education
train.boxplot(column='CoapplicantIncome', by='Education')
plt.suptitle('')
plt.show()

# Look at the loan amount distribution
plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.histplot(df['LoanAmount'])

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,10))
plt.show()

train.boxplot(column='LoanAmount', by='Loan_Status')
plt.suptitle('')
plt.show()

# # loan amount term distribution
plt.figure(1)
plt.subplot(121)
df = train['Loan_Amount_Term'].dropna()
sns.histplot(df)

plt.subplot(122)
train['Loan_Amount_Term'].plot.box(figsize=(16,10))

train.boxplot(column='Loan_Amount_Term', by='Loan_Status')
plt.suptitle('')
plt.show()


# Bivariate analysis


# Relation between loan status and gender in a stacked bar plot
plt.xlabel('Gender')
plt.ylabel('Percentage')
Gender = pd.crosstab(train['Gender'], train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(4,4))
print(Gender)
plt.show()

# Relation between loan status and Education
plt.xlabel('Education')
plt.ylabel('Percentage')
Education = pd.crosstab(train['Education'], train['Loan_Status'])
Education.div(Education.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(10,6))
print(Education)
plt.show()

# Relation between loan status and property area
plt.xlabel('Property_Area')
plt.ylabel('Percentage')
Property_Area = pd.crosstab(train['Property_Area'], train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(12,10))
print(Property_Area)
plt.show()

# Relationship between loan status and credit history
plt.xlabel('Credit_History')
plt.ylabel('Percentage')
Credit_History = pd.crosstab(train['Credit_History'], train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(14,12))
print(Credit_History)
plt.show()

# finding the mean income of people for which the loan was approved vs disapproved
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
plt.show()

# making histogram with categories(bins)
# making the intervals(bins)
bins = [0, 2500, 4000, 6000, 81000]
groups = ['low', 'Average', 'High', 'Very High']
df = train.dropna()
train['Income_bin'] = pd.cut(df['ApplicantIncome'], bins, labels=groups)

plt.xlabel('ApplicantIncome')
plt.ylabel('Percentage')
Income_bin = pd.crosstab(train['Income_bin'], train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(14,12))
print(Income_bin)
plt.show()

#  Finding the total income of applicant and coapplicant
train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']

# find relationship between the total income and the loan status
bins = [0, 2500, 4000, 6000, 81000]
groups = ['Low', 'Average', 'High', 'Very High']
train['TotalIncome_bin'] = pd.cut(train['TotalIncome'], bins, labels=groups)

plt.xlabel('TotalIncome')
plt.ylabel('Percentage')
TotalIncome = pd.crosstab(train['TotalIncome_bin'], train['Loan_Status'])
TotalIncome.div(TotalIncome.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(2,2))
print(TotalIncome)
plt.show()

# Relationship between the loan amount and the loan status
bins = [0, 100, 200, 700]
groups = ['Low', 'Average', 'High']
train['LoanAmount_bin'] = pd.cut(train['LoanAmount'], bins, labels=groups)

plt.xlabel('LoanAmount')
plt.ylabel('Percentage')
LoanAmount = pd.crosstab(train['LoanAmount_bin'], train['Loan_Status'])
LoanAmount.div(LoanAmount.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(16,12))
print(LoanAmount)
plt.show()


# Remove the bins
train = train.drop(['Income_bin','TotalIncome_bin','LoanAmount_bin', 'TotalIncome'], axis=1)

# Change the 3+ in Dependents to 3, so it is a numerical variable
train['Dependents'].replace('3+',3, inplace=True)
test['Dependents'].replace('3+', 3, inplace=True)

# Change loan status from yes/no to 1/0
train['Loan_Status'].replace('Y', 1, inplace=True)
train['Loan_Status'].replace('N', 0, inplace=True)

# Using a heat map to see the correlation
# Darker colours mean more correlation
# The whole train file is being correlation between each other

matrix = train.corr()
fig, axes = plt.subplots(figsize=(14,10))
sns.heatmap(matrix, yticklabels='auto', vmax=1, square=True, cmap='OrRd', annot=True, cbar_kws={'label': 'Correlation'})
plt.show()

# List of all the missing values
train.isnull().sum()

# Fill using the mode of the features
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

# fill the missing values in the loan amount term with the mode
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

# when dealing with numerical values, fill the gaps with the mean or median
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

# fill using the mode of the features
test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)

# for numerical val, use median/mean
test['LoanAmount'].fillna(test['LoanAmount'].mean(), inplace=True)

# plot a distribution for loan amount
sns.histplot(train['LoanAmount'])
plt.show()

train['LoanAmount'].hist(bins=20)
plt.show()

# to remove skewness, do a log transformation
train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
plt.show()

sns.displot(train['LoanAmount_log'], kde=True)
plt.show()

test['LoanAmount_log'] = np.log(test['LoanAmount'])
test['LoanAmount_log'].hist(bins=20)

sns.displot(test['LoanAmount_log'], kde=True)
plt.show()

# Creating total income
train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
train[['TotalIncome']].head()

test['TotalIncome'] = test['ApplicantIncome'] + test['CoapplicantIncome']
test['TotalIncome'].head()

# check distribution for train total income
sns.distplot(train['TotalIncome'], kde=True)
plt.show()

# take the log of total income
train['TotalIncome_log'] = np.log(train['TotalIncome'])
sns.distplot(train['TotalIncome_log'])
plt.show()

# check distribution for test total income
sns.distplot(test['TotalIncome'])
plt.show()

# take the log of total income
test['TotalIncome_log'] = np.log(test['TotalIncome'])
sns.distplot(test['TotalIncome_log'])
plt.show()

# Create the EMI feature - monthly amount paid by the applicant
train['EMI'] = train['LoanAmount']/train['Loan_Amount_Term']
test['EMI'] = test['LoanAmount']/test['Loan_Amount_Term']

# distribution for emi
sns.distplot(train['EMI'])
plt.show()

sns.distplot(test['EMI'])
plt.show()

# Create balance income - balance left after loan has been repaid
train['Balance_Income'] = train['TotalIncome']-train['EMI']*1000
test['Balance_Income'] = test['TotalIncome']-test['EMI']*1000

# dropping old variables
train = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

# model building

# drop loan id as it has no impact on loan status
train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

# drop loan status from the dataset and save it in another dataset
X = train.drop('Loan_Status', 1)

# save loan status in y
y = train['Loan_Status']

# Making dummy variables, changes headings into binary - Gender changes to Gender_Male and Gender_Female
# Gender_Male will have a value of 0 and Gender_Female will have a value of 1
# It does the same with the rest of the headings
X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# divide train dataset into train and validation
# train the model on train, and make predictions on the validation dataset
# as the test size is 30% the train size will be 70% of the data
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=1)

# logistic regression
logistic_model = LogisticRegression(random_state=1)
logistic_model.fit(x_train, y_train)

# Predict loan status for validation set
pred_cv_logistic = logistic_model.predict(x_cv)

# calculate how accurate our predictions are by calculating the accuracy
score_logistic = accuracy_score(pred_cv_logistic, y_cv)*100

# predictions for the test data set
pred_test_logistic = logistic_model.predict(test)

# Using decision tree
tree_model = DecisionTreeClassifier(random_state=1)

# fitting the model
tree_model.fit(x_train, y_train)

pred_cv_tree = tree_model.predict(x_cv)
score_tree = accuracy_score(pred_cv_tree, y_cv)*100

# predictions for the test dataset
pred_test_tree = tree_model.predict(test)


# random forest
# max depth decides max depth of the tree, n_estimators decides number of trees used in a random forest model
forest_model = RandomForestClassifier(random_state=1, max_depth=10, n_estimators=50)
forest_model.fit(x_train, y_train)

pred_cv_forest = forest_model.predict(x_cv)
score_forest = accuracy_score(pred_cv_forest, y_cv)*100

# predictions for the test dataset
pred_test_forest = forest_model.predict(test)

# Using random forest with grid search
# set max depth from 1-20 with interval of 2. Set nestimators 1-200 with interval of 20
paramgrid = {'max_depth': list(range(1,20,2)), 'n_estimators': list(range(1,200,20))}
grid_search = GridSearchCV(RandomForestClassifier(random_state=1), paramgrid)

# fit the grid search model
grid_search.fit(x_train, y_train)

# find out the optimised value
grid_search.best_estimator_

# build the model using the optimised values
grid_forest_model = RandomForestClassifier(random_state=1, max_depth=7, n_estimators=41)
grid_forest_model.fit(x_train, y_train)

pred_grid_forest = grid_forest_model.predict(x_cv)
score_grid_forest = accuracy_score(pred_grid_forest, y_cv)*100

# predictions for the test dataset
pred_grid_forest_test = grid_forest_model.predict(test)


# using xgboost model
xgb_model = XGBClassifier(n_estimators=50, max_depth=4)
xgb_model.fit(x_train, y_train)

pred_xgb = xgb_model.predict(x_cv)
score_xgb = accuracy_score(pred_xgb, y_cv)*100

# find the important feature
importances = pd.Series(forest_model.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(16,12))
plt.show()
