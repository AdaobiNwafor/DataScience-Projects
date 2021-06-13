import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import scipy as sp
import sklearn
import seaborn as sns
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier


# importing the data sets
# validation
valid = pd.read_csv("C:\\Users\\adaob\\PycharmProjects\\pythonProject2\\Titanic\\test.csv")
valid.head()

# train and test sets
raw = pd.read_csv("C:\\Users\\adaob\\PycharmProjects\\pythonProject2\\Titanic\\train.csv")
raw.head()

# create a copy
data1 = raw.copy(deep=True)

data_cleaner = [data1, valid]

# previewing the data
raw.info()

raw.sample(10)

# finding null values
print(f'Train columns with null values:\n {raw.isnull().sum()}')
print('--'*20)

print(f'Test/Val columns with null values:\n {valid.isnull().sum()}')


# understand the data
raw.describe(include='all')

# complete or delete missing values in train/test
for dataset in data_cleaner:
    # completing the missing values with the median
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

    # complete embarked column with the mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

    # complete the missing fare values with the median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

# delete the cabin, passenger id and ticket column as they do not help with the data prediction
# drop_column = ['PassengerId', 'Cabin', 'Ticket', 'SibSp', 'Parch']
drop_column = ['SibSp', 'Parch']
data1.drop(drop_column, axis=1, inplace=True)

# find out how many null values after data cleaned
print(f'The number of missing values in train data: {data1.isnull().sum()}')
print('--'*20)
print(f'The number of missing values in the test/val data: {valid.isnull().sum()}')

# continuous variable bins using qcut
for dataset in data_cleaner:

    # splitting the names of people into their titles
    dataset['Title'] = dataset['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4)
    dataset['Agebin'] = pd.cut(dataset['Age'].astype(int), 5)


# cleaning up the rare title names
stat_min = 10
title_names = (data1['Title'].value_counts() < stat_min)

data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print('---'*10)

# coding categorical data
label = LabelEncoder()
for dataset in data_cleaner:

    dataset['Sex_code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_code'] = label.fit_transform(dataset['Title'])
    dataset['Agebin_code'] = label.fit_transform(dataset['Agebin'])
    dataset['Farebin_code'] = label.fit_transform(dataset['Farebin'])

# define the y variable (target variable)

Target = ['Survived']

# define the x variables for original features aka feature selection
data1_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'Age', 'Fare']
data1_x_calc = ['Sex_code', 'Pclass', 'Embarked_code', 'Title_code', 'Age', 'Fare']
data1_xy = Target + data1_x

print(f'Original X Y: {data1_xy}\n')

# define the x variables for original with bin features to remove continuous variables
data1_x_bin = ['Sex_code', 'Pclass', 'Embarked_code', 'Title_code', 'Agebin_code', 'Farebin_code']
data1_xy_bin = Target + data1_x_bin
print(f'Bin X Y: {data1_xy_bin}\n')


# define x and y variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print(f'Dummy X Y: {data1_xy_dummy}\n')


# splitting the train dataset into train and test, using a 75/25 split

train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state=0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target], random_state=0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state=0)

print(f'Data1 shape: {data1.shape}')
print(f'Train1 shape: {train1_x.shape}')
print(f'Test1 shape: {test1_x.shape}')

train1_x_bin.head()

# performing exploratory analysis with statistics

# discrete variable corellation by survival using group by pivot table

for x in data1_x:
    if data1[x].dtype != 'float64':
        print(f'Survival correlation by: {x}')
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('---'*10)


# using crosstabs
print(pd.crosstab(data1['Title'], data1[Target[0]]))

# graph distribution of quantitative data

plt.figure(figsize=(14,12))

plt.subplot(231)
plt.boxplot(x=data1['Fare'], showmeans=True, meanline=True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(x=data1['Age'], showmeans=True, meanline=True)
plt.title('Age Boxplot')
plt.ylabel('Age (years)')

plt.subplot(233)
plt.hist(x=[data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], stacked=True,
         color=['g', 'r'], label= ['Survived', 'Dead'])
plt.title('Fare histogram by survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of passengers')
plt.legend()

plt.subplot(234)
plt.hist(x=[data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']],
         stacked=True, color=['r', 'b'], label=['Survived', 'Dead'])
plt.title('Age histogram by survival')
plt.xlabel('Age (years)')
plt.ylabel('# if passengers')
plt.legend()

# seaborn for multi variable statistics

fig, saxis = plt.subplots(2, 2, figsize=(14,12))
sns.barplot(x='Embarked', y='Survived', data=data1, ax=saxis[0,0])
sns.barplot(x='Pclass', y='Survived', data=data1, ax=saxis[0,1])

sns.pointplot(x='Farebin', y='Survived', data=data1, ax=saxis[1,0])
sns.pointplot(x='Agebin', y='Survived', data=data1, ax=saxis[1,1])

# graph distribution of qualitative data

# we know class mattered, so compare class and a second feature

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(14,12))

sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=data1, ax=axis1)
axis1.set_title('Pclass Vs Fare Survival comparison')

sns.violinplot(x='Pclass', y='Age', hue='Survived', data=data1, split=True, ax=axis2)
axis2.set_title('Pclass Vs Age Survival comparison')

# graph distribution pf qualitative data: Sex
# we know sex mattered so compare this with another feature

fig, qaxis = plt.subplots(1, 2, figsize=(14,12))

sns.barplot(x='Sex', y='Survived', hue='Embarked', data=data1, ax=qaxis[0])
qaxis[0].set_title('Sex vs Embarked Survival comparison')

sns.barplot(x='Sex', y='Survived', hue='Pclass', data=data1, ax=qaxis[1])
qaxis[1].set_title('Sex vs Pclass survival comparison')

# more side by side comparisons

fig, (maxis) = plt.subplots(1, figsize=(10,6))#

sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=data1, palette={'male': 'blue', 'female': 'pink'},
              markers=['*', 'o'], linestyles=['-', '--'], axis=maxis)
maxis.set_title('Pclass vs Male and Female survival comparison')

# How does the embark fort factor with class, sex and survival compare

e = sns.FacetGrid(data1, col='Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95, palette='deep')
e.add_legend()

# plot distributions of age of passengers who survived
a = sns.FacetGrid(data1, hue='Survived', aspect=3)
a.map(sns.kdeplot, 'Age', shade=True)
a.set(xlim=(0, data1['Age'].max()))
a.add_legend()

# histogram comparison of sex, class and age by survival
b = sns.FacetGrid(data1, row='Sex', col='Pclass', hue='Survived')
b.map(plt.hist, 'Age', alpha=0.75)
b.add_legend()

# pair plots of the entire dataset
pp = sns.pairplot(data1, hue='Survived', palette='deep', size=1.2, diag_kind='kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
pp.set(xticklabels=[])


# Modelling the data

#  Selecting which Machine learning algorithm to use (MLA) using ensemble,
#  gaussian processes, GLM, Naive Bayes, SVM, Trees, Discriminant Analysis and xgboost

MLA = [
    # ensemble
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Process
    gaussian_process.GaussianProcessClassifier(),

    # GLM (Generalised Linear Models)
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    # Naive Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest neighbour
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #xgboost
    XGBClassifier()
]

# split the data in cross validation with the splitter class
# this is an alternative to train test split

# intentionally leaving out 10%
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)

# create a table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA train accuracy mean', 'MLA test accuracy mean', 'MLA test accuracy 3*STD', 'MLA time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

# create a table to compare the MLA predictions
MLA_predicts = data1[Target]

# index through MLA and save the performance to a table
row_index = 0
for alg in MLA:

    # set the name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    # score the model with cross validation (cv)
    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv=cv_split, return_train_score=True)

    MLA_compare.loc[row_index, 'MLA time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA train accuracy mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA test accuracy mean'] = cv_results['test_score'].mean()

    # if this is a non bias random sample then +/-3 standard deviations from the mean should
    # capture 99.7% of the subsets

    MLA_compare.loc[row_index, 'MLA test accuracy 3*STD'] = cv_results['test_score'].std()*3

    # save MLA predictions
    alg.fit(data1[data1_x_bin], data1[Target])
    MLA_predicts[MLA_name] = alg.predict(data1[data1_x_bin])

    row_index += 1

# print and sort the table
MLA_compare.sort_values(by=['MLA test accuracy mean'], ascending=False, inplace=True)
MLA_compare


# barplot
sns.barplot(x='MLA test accuracy mean', y='MLA Name', data=MLA_compare, color='m')
plt.title('Machine Learning Algorithm Accuracy Score\n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')

# base model using decision trees
dtree = tree.DecisionTreeClassifier(random_state=0)
base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv=cv_split, return_train_score=True)
dtree.fit(data1[data1_x_bin], data1[Target])

print(f'Before Decision Tree parameters: {dtree.get_params()}\n')
print(f'Before Decision Tree training with bin score mean: {base_results["train_score"].mean()*100:.2f}\n')
print(f'Before Decision Tree test with bin score mean: {base_results["test_score"].mean()*100:.2f}\n')
print(f'Before Decision Tree test with bin score 3*std: +/- {base_results["test_score"].std()*100*3}')
print('--'*10)

param_grid = {'criterion':['gini', 'entropy'], 'max_depth': [2,4,6,8,10,None], 'random_state': [0]}

# choose the best model with grid search
tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc', cv=cv_split, return_train_score=True)
tune_model.fit(data1[data1_x_bin], data1[Target])

# print the keys and parameters to know what to use
# print(tune_model.cv_results_.keys())
# print(tune_model.cv_results_['params'])

print(f'After Decision Tree Parameters: {tune_model.best_params_}\n')
print(f'After Decision Tree training with bin score mean: {tune_model.cv_results_["mean_train_score"][tune_model.best_index_]*100:.2f}\n')
print(f'After Decision Tree test with bin score mean: {tune_model.cv_results_["mean_test_score"][tune_model.best_index_]*100:.2f}\n')
print(f'After Decision Tree with bin score 3*STD: +/- {tune_model.cv_results_["std_test_score"][tune_model.best_index_]*100*3:.2f}\n')
print('--'*10)




