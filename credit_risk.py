import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#pyt!pip install pandas

df_loan=pd.read_csv("loan.csv")
df_loan

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns',None)

import warnings
warnings.filterwarnings('ignore')

df_loan.shape

# Compulsory drop of ID type of Features\n",
# 1) ID type features simply act as row identifiers when data get built into the database\n",
# 2) ID can falsely become the best predictor when it ideally should not be a driving factor for the outcome\n",
df_loan=df_loan.drop(['id','member_id'],axis=1)
  

df_loan.dtypes

# Creating a Frequency Table in Pandas for each categorical\n",
# dataframe['categorical feature'].value_counts()  --- value_counts() function 
#will show you which all values the categorical \n",
# feature will take and also how many observations are there for each of them"
   
#Analyzing loan_status variable for getting a better understanding of the categories and the counts\n",
df_loan['loan_status'].value_counts()

# Define the Dependent Variable "
df_loan['target']=np.where(df_loan['loan_status'].isin(['Default','Charged Off','Does not meet the credit policy. Status:Charged Off']),1,0)
#df_loan['target']

# Drop the loan_status variable\n",
df_loan=df_loan.drop(['loan_status'],axis=1)

# Calculate the default rate or event rate in the data"
df_loan['target'].mean()

"# Missing Value Analysis"
df_loan.isnull().mean()

#data['open_acc_6m_Rank']=pd.qcut(data['open_acc_6m'].rank(method='first').values,10,duplicates='drop').codes+1"
# Selecting all rows and only those columns where the missing value percentage is <=25%\n",
data=df_loan.loc[:,df_loan.isnull().mean()<=0.25]

data.shape   
# 22 Features dropped Due to more than 25% Missing Values"

# Creating a dataset with only dependent and independent\n",
Y=data[['target']]
X=data.drop(['target'],axis=1)
 

X.dtypes
char=X.select_dtypes(include='object')
num=X.select_dtypes(include='number')

# "# We have 19 Categorical features\n",
char.shape
num.shape
num.isnull().mean()

def outlier_cap(x):
    x=x.clip(lower=x.quantile(0.01))
    x=x.clip(upper=x.quantile(0.99))
    return(x)

num=num.apply(lambda x : outlier_cap(x))

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
num_1=pd.DataFrame(imputer.fit_transform(num),index=num.index,columns=num.columns)

num_1.isnull().mean()  

# Replace the missing valus in Categorical using the Mode or the Most Frequent Strategy\n",
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
char_1=pd.DataFrame(imputer.fit_transform(char),index=char.index,columns=char.columns)


char_1=char.fillna(0)
char_1.isnull().mean()
# To avoid discrimation an enable fair lending, we are not using any title variables are risk identifier factors\n",
char_1=char_1.drop(['url','zip_code','emp_title','issue_d','addr_state','title','sub_grade','last_credit_pull_d','earliest_cr_line','last_pymnt_d'],axis=1)

# 10 Categorical Features droppping based on Fair Lending Considerations"
# Joining back the target  variable and exporting the pivot table for a BiVariate Analysis\n",
categorical_variable_chk=pd.concat([Y,char_1],axis=1,join='inner')
categorical_variable_chk.to_csv('loan.csv')

from sklearn.preprocessing import KBinsDiscretizer
discrete=KBinsDiscretizer(n_bins=10,encode='ordinal', strategy='quantile')
num_binned=pd.DataFrame(discrete.fit_transform(num_1),index=num_1.index, columns=num_1.columns).add_suffix('_Rank')
num_binned.head()

#Check if the features show a slope at all\n",
#If they do, then do you see some deciles below the population average and some higher than population average?\n",
#If that is the case then the slope will be strong\n",
#Conclusion: A strong slope is indicative of the features' ability to discriminate the event from non event\n",
#            making it a good predictor\n",
   
#percentage_income_goesinto_intallments=Insallment/annual_inc (Derived Variables/Feature Engineering)\n",
    
X_bin_combined=pd.concat([Y,num_binned],axis=1,join='inner')
    
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean
for col in (X_bin_combined.columns):
    plt.figure()
    sns.barplot(x=col, y="target",data=X_bin_combined, estimator=mean )
    plt.show()

#https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

# Select K Best for Numerical Features\n",
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=15)
X_new = selector.fit_transform(num_1, Y)
# Get columns to keep and create new dataframe with those only\n",
cols = selector.get_support(indices=True)
select_features_df_num = num_1.iloc[:,cols]

select_features_df_num.loc[:0]

select_features_df_num.dtypes

import matplotlib.pyplot as plt
import seaborn as sns
X_char_merged=pd.concat([Y,char_1],axis=1,join='inner')
  
from numpy import mean
for col in (char_1.columns):
    plt.figure()
    sns.barplot(x=col, y="target",data=X_char_merged, estimator=mean )
    plt.show()
   
# Steps to regroup categorical features in order to achieve linear discrimination\n",
# These levels to group have been discovered through the pivot analysis\n",
char_1['verified_1']=np.where(char_1['verification_status'].isin(['Source Verified','Verified']),'Verified',char_1['verification_status'])
char_1['purpose_1']=np.where(char_1['purpose'].isin(['other','medical','vacation','debt_consolidation','car','major_purchase','home_improvement','credit_card']),'others',char['purpose'])           
char_1['purpose_1']=np.where(char_1['purpose'].isin(['other','medical','vacation','debt_consolidation','car','major_purchase','home_improvement','credit_card']),'others',char['purpose'])           
char_1['home_own_1']=np.where(char_1['home_ownership'].isin(['OWN','MORTGAGE']),'OWN',char_1['home_ownership'])
char_1['term_1']=np.where(char_1['term'].isin(['60 months']),'60','30')

char_2=char_1.loc[:,['purpose_1','verified_1','home_own_1','term_1','grade','emp_length','pymnt_plan','application_type']]

check=pd.concat([Y,char_2],axis=1,join="inner")
ax=sns.barplot(x="purpose_1", y="target",data=check, estimator=mean )

# Create dummy features with n-1 levels\n",
X_char_dum = pd.get_dummies(char_2, drop_first = True)

X_char_dum.shape

# Select K Best for Categorical Features\n",
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=25)
X_new_1 = selector.fit_transform(X_char_dum, Y)
# Get columns to keep and create new dataframe with those only\n",
cols = selector.get_support(indices=True)
select_features_df_char = X_char_dum.iloc[:,cols]

select_features_df_char.loc[:0]
select_features_df_char.dtypes

num_additional=num_1.loc[:,['int_rate','inq_last_6mths']]
num_additional.head()

# Bringing it together\n",
X_all=pd.concat([select_features_df_char,select_features_df_num,num_additional],axis=1,join="inner")

Y['target'].value_counts()   
Y.mean()
# Split the data between Train and Testing datasets\n",
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_all, Y, test_size=0.3, random_state=42)

X_train['revol_bal'].describe() 
X_test['revol_bal'].describe()
X_all['revol_bal'].describe()
y_train.mean()
y_test.mean() 

# Non Linearity in feature relationships are observed which makes tree methods a good choice\n",
# There are few options to consider among tree methods\n",
# White Box (Completely Explainable Set of Rules) - Decision Tree\n",
# Ensemble Methods - Random Forest (With Bagging)\n",
# Ensemble Methods - GBM/XGBoost (Boosting)"

DecisionTreeClassifier(random_state=0)
# Building a Decision Tree Model\n",
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(random_state=0)
dtree.fit(X_train,y_train)

from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
plt.figure(figsize=[50,10])
tree.plot_tree(dtree,filled=True,fontsize=20,rounded=True,feature_names=X_all.columns)
plt.show()

RandomForestClassifier(random_state=0)
# Building a Random Forest Model\n",
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)

import pandas as pd
feature_importances=pd.DataFrame(rf.feature_importances_,
                                    index=X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances

# Building a Gradient Boosting Model\n",
from sklearn.ensemble import GradientBoostingClassifier
GradientBoostingClassifier(random_state=0)
clf=GradientBoostingClassifier(random_state=0)
clf.fit(X_train,y_train) 

import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]

# Building a XGBoost Model\n",
from xgboost import XGBClassifier
xgb=XGBClassifier(random_state=0)
xgb.fit(X_train,y_train)

# Model Evaluation\n",
y_pred=clf.predict(X_test)
y_pred_tree=dtree.predict(X_test)
y_pred_rf=rf.predict(X_test)
y_pred_xgb=xgb.predict(X_test)

# Model Evaluation\n",
y_pred=clf.predict(X_test)
y_pred_tree=dtree.predict(X_test)
y_pred_rf=rf.predict(X_test)
y_pred_xgb=xgb.predict(X_test)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
y_pred_tree=dtree.predict(X_test)
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_tree))
print("Precision:",metrics.precision_score(y_test,y_pred_tree))
print("Recall:",metrics.recall_score(y_test,y_pred_tree))
print("f1_score :",metrics.f1_score(y_test,y_pred_tree))

metrics.plot_confusion_matrix(dtree,X_all,Y)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rf)),
print("Precision",metrics.precision_score(y_test,y_pred_rf)),
print("Recall",metrics.recall_score(y_test,y_pred_rf)),
print("f1_score",metrics.f1_score(y_test,y_pred_rf))

metrics.plot_confusion_matrix(rf,X_all,Y)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision",metrics.precision_score(y_test,y_pred))
print("Recall",metrics.recall_score(y_test,y_pred))
print("f1_score",metrics.f1_score(y_test,y_pred))

metrics.plot_confusion_matrix(clf,X_all,Y)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_xgb)),
print("Precision",metrics.precision_score(y_test,y_pred_xgb)),
print("Recall",metrics.recall_score(y_test,y_pred_xgb)),
print("f1_score",metrics.f1_score(y_test,y_pred_xgb))

metrics.plot_confusion_matrix(xgb,X_all,Y)
# Lorenz Curve"
#Decsion Tree Lorenz Curve"
y_pred_prob = dtree.predict_proba(X_all)[:, 1]
df_loan['y_pred_P']=pd.DataFrame(y_pred_prob)
df_loan['P_Rank_RF']=pd.qcut(df_loan['y_pred_P'].rank(method='first').values,10,duplicates='drop').codes+1
rank_df=df_loan.groupby('P_Rank_RF')['target'].agg(['count','mean'])
rank_df=pd.DataFrame(rank_df)
sorted_rank_df=rank_df.sort_values(by='P_Rank_RF',ascending=False)
sorted_rank_df['N_events']=rank_df['count']*rank_df['mean']
sorted_rank_df['cum_events']=sorted_rank_df['N_events'].cumsum()
sorted_rank_df['event_cap']=sorted_rank_df['N_events']/max(sorted_rank_df['N_events'].cumsum())
sorted_rank_df['cum_event_cap']=sorted_rank_df['event_cap'].cumsum()
sorted_rank_df['random_cap']=sorted_rank_df['count']/max(sorted_rank_df['count'].cumsum())
sorted_rank_df['cum_random_cap']=sorted_rank_df['random_cap'].cumsum()
sorted_reindexed=sorted_rank_df.reset_index()
sorted_reindexed['decile']=sorted_reindexed.index+1
sorted_reindexed['lift_over_random']=sorted_reindexed['cum_event_cap']/sorted_reindexed['cum_random_cap']
sorted_reindexed

ax = sns.lineplot( x="decile", y="lift_over_random", data=sorted_reindexed)
ax = sns.lineplot( x="decile", y="random_cap", data=sorted_reindexed)

df_loan.groupby('P_Rank_RF')['y_pred_P'].agg(['min','max'])

y_pred_prob = xgb.predict_proba(X_all)[:, 1]
df_loan['y_pred_P']=pd.DataFrame(y_pred_prob)
df_loan['P_Rank_RF']=pd.qcut(df_loan['y_pred_P'].rank(method='first').values,10,duplicates='drop').codes+1
rank_df=df_loan.groupby('P_Rank_RF')['target'].agg(['count','mean'])
rank_df=pd.DataFrame(rank_df)
sorted_rank_df=rank_df.sort_values(by='P_Rank_RF',ascending=False)
sorted_rank_df['N_events']=rank_df['count']*rank_df['mean']
sorted_rank_df['cum_events']=sorted_rank_df['N_events'].cumsum()
sorted_rank_df['event_cap']=sorted_rank_df['N_events']/max(sorted_rank_df['N_events'].cumsum())
sorted_rank_df['cum_event_cap']=sorted_rank_df['event_cap'].cumsum()
sorted_rank_df['random_cap']=sorted_rank_df['count']/max(sorted_rank_df['count'].cumsum())
sorted_rank_df['cum_random_cap']=sorted_rank_df['random_cap'].cumsum()
sorted_reindexed=sorted_rank_df.reset_index()
sorted_reindexed['decile']=sorted_reindexed.index+1
sorted_reindexed['lift_over_random']=sorted_reindexed['cum_event_cap']/sorted_reindexed['cum_random_cap']
sorted_reindexed

ax = sns.lineplot( x="decile", y="lift_over_random", data=sorted_reindexed)
ax = sns.lineplot( x="decile", y="random_cap", data=sorted_reindexed)

x=[15, 18, 21, 22, 26, 28, 28]
df_loan=pd.DataFrame(x)
df_loan.quantile(0.5)
#df_loan.var()t

# Project Conclusion :- 
# Create a decision tree and show it to stakeholders. This will create awareness about the kind of rules that a Tree might 
# come up with
# Now speak about the benefit of ensembles and how they achieve a higher degress of robustness than a single tree
# Now is the time to show that the ensemble is as good if not better than the decison tree
# This way you can first sell the idea of a tree based solution and then sell the final product which is an ensemble of the
# idea you have already sold(decision tree)

