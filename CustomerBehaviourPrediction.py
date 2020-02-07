# Import Libraries. pandas, numpy, matplotlib, serborn etc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# This is for inline plotting in juypyer notebook
get_ipython().magic('matplotlib inline')

# Read in the advertising.csv file and set it to a data frame called ad_data.
ad= pd.read_csv('advertising.csv')

###########################################################################################################

#Exploratory Data Analysis
sns.set_style('white')
sns.set_context('notebook')

#Summary with respect to clicked on ad
sns.pairplot(ad, hue='Clicked on Ad', palette='bwr')

#Click on Ad features based on Sex
plt.figure(figsize=(10,6))
sns.countplot(x='Clicked on Ad',data=ad,hue='Male',palette='coolwarm')

#Distribution of top 12 country's ad clicks based on Sex
plt.figure(figsize=(15,6))
sns.countplot(x='Country',data=ad[ad['Clicked on Ad']==1],order=ad[ad['Clicked on Ad']==1]['Country'].value_counts().index[:12],hue='Male',
              palette='viridis')
plt.title('Ad clicked country distribution')
plt.tight_layout()

#Changing the datetime object
ad['Timestamp']=pd.to_datetime(ad['Timestamp'])

#Introduce new columns Hour,Day of Week, Date, Month from timestamp
ad['Hour']=ad['Timestamp'].apply(lambda time : time.hour)
ad['DayofWeek'] = ad['Timestamp'].apply(lambda time : time.dayofweek)
ad['Month'] = ad['Timestamp'].apply(lambda time : time.month)
ad['Date'] = ad['Timestamp'].apply(lambda t : t.date())

#Hourly distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='Hour',data=ad[ad['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked hourly distribution')

#Daily distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='DayofWeek',data=ad[ad['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked daily distribution')

#Monthly distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='Month',data=ad[ad['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked monthly distribution')

#group by date
plt.figure(figsize=(15,6))
ad[ad['Clicked on Ad']==1].groupby('Date').count()['Clicked on Ad'].plot()
plt.title('Date wise distribution of Ad clicks')
plt.tight_layout()


#Top Ad clicked on specific date
ad[ad['Clicked on Ad']==1]['Date'].value_counts().head(5)

ad['Ad Topic Line'].nunique()
#All 1000 ad topics are different and hence it is difficult to feed to the model. (TF_IDF?)

#Age distribution
plt.figure(figsize=(10,6))
sns.distplot(ad['Age'],kde=False,bins=40)

#Age distribution
plt.figure(figsize=(10,6))
sns.swarmplot(x=ad['Clicked on Ad'],y= ad['Age'],data=ad,palette='coolwarm')
plt.title('Age wise distribution of Ad clicks')

#Daily internet usage and daily time spent on site based on age
fig, axes = plt.subplots(figsize=(10, 6))
ax = sns.kdeplot(ad['Daily Time Spent on Site'], ad['Age'], cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(ad['Daily Internet Usage'],ad['Age'] ,cmap="Blues", shade=True, shade_lowest=False)
ax.set_xlabel('Time')
ax.text(20, 20, "Daily Time Spent on Site", size=16, color='r')
ax.text(200, 60, "Daily Internet Usage", size=16, color='b')


#distribution who clicked on Ad based on area income of sex
plt.figure(figsize=(10,6))
sns.violinplot(x=ad['Male'],y=ad['Area Income'],data=ad,palette='viridis',hue='Clicked on Ad')
plt.title('Clicked on Ad distribution based on area distribution')

#country value as dummies
country= pd.get_dummies(ad['Country'],drop_first=True)

#drop the columns not required for building a model
ad.drop(['City','Country','Timestamp','Date'],axis=1,inplace=True)

#join the dummy values
ad = pd.concat([ad,country],axis=1)

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ad.columns = [c.replace(' ', '_') for c in ad.columns]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(ad.Ad_Topic_Line)
e_atl = le.transform(ad.Ad_Topic_Line)
ad.Ad_Topic_Line = e_atl

ad.columns = [c.replace('_', ' ') for c in ad.columns]


########################################################################################################
#LOGISTIC REGRESSION MODEL

from sklearn.model_selection import train_test_split
X= ad.drop('Clicked on Ad',axis=1)
y= ad['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train,y_train)

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100,1000]}

grid_log= GridSearchCV(LogisticRegression(),param_grid,refit=True, verbose=2)

grid_log.fit(X_train,y_train)

grid_log.best_estimator_

pred_log= grid_log.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(confusion_matrix(y_test,pred_log))
print(classification_report(y_test,pred_log))

"""
[[102   3]
 [ 10  85]]
              precision    recall  f1-score   support

           0       0.91      0.97      0.94       105
           1       0.97      0.89      0.93        95

   micro avg       0.94      0.94      0.94       200
   macro avg       0.94      0.93      0.93       200
weighted avg       0.94      0.94      0.93       200
"""
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, grid_log.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, grid_log.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
########################################################################################################
#Support Vector Model

from sklearn.svm import SVC
svc= SVC(gamma='scale')

svc.fit(X_train,y_train)

param_grid = {'C': [0.1,1,10,100,1000,5000]}

grid_svc= GridSearchCV(SVC(gamma='scale',probability=True),param_grid,refit=True,verbose=2)

grid_svc.fit(X_train,y_train)

grid_svc.best_estimator_

pred_svc= grid_svc.predict(X_test)
print(confusion_matrix(y_test,pred_svc))
print(classification_report(y_test,pred_svc))

"""
[[101   4]
 [  9  86]]
              precision    recall  f1-score   support

           0       0.92      0.96      0.94       105
           1       0.96      0.91      0.93        95

   micro avg       0.94      0.94      0.94       200
   macro avg       0.94      0.93      0.93       200
weighted avg       0.94      0.94      0.93       200
"""
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, grid_svc.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, grid_svc.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

########################################################################################################

#let's first scale the variables
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()

scaler.fit(ad.drop('Clicked on Ad',axis=1))

scaled_features= scaler.transform(ad.drop('Clicked on Ad',axis=1))

#Changing it from numpy array to pandas dataframe
train_scaled = pd.DataFrame(scaled_features,columns=ad.columns.drop('Clicked on Ad'))
train_scaled.head()

X_train, X_test, y_train, y_test = train_test_split(train_scaled,ad['Clicked on Ad'],test_size=0.20,random_state=101)

from sklearn.neighbors import KNeighborsClassifier
error_rate=[]

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K-value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn= KNeighborsClassifier(n_neighbors=40)
knn.fit(X_train,y_train)

pred_knn=knn.predict(X_test)
print(confusion_matrix(y_test,pred_knn))
print(classification_report(y_test,pred_knn))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, knn.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, knn.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

"""
[[99  6]
 [ 6 89]]
              precision    recall  f1-score   support

           0       0.94      0.94      0.94       105
           1       0.94      0.94      0.94        95

   micro avg       0.94      0.94      0.94       200
   macro avg       0.94      0.94      0.94       200
weighted avg       0.94      0.94      0.94       200
"""

########################################################################################################

#Either KNN or SVM
