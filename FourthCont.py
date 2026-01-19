import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import pickle
from sklearn.linear_model import LassoCV

df=pd.read_csv('Algerian_forest_fires_CLEANED_dataset.csv',header=0)
print(df.head())

df['Classes']=np.where(df['Classes'].str.contains('not fire'),0,1)
print(df['Classes'].value_counts())

# INDEPENDENT AND DEPENDENT FEATURES

X=df.drop('FWI',axis=1)
Y=df['FWI']
print(X.head())

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# Feature Selection using Correlation Matrix
plt.figure(figsize=(12,10))
corr=X_train.corr()
sns.heatmap(corr,annot=True)
plt.show()

## now we will try to identify those features which are highly correlated in order to remove them
def correlation(dataset,threshold):
    col_corr=set() # Set of all the names of correlated columns
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features=correlation(X_train,0.85)
X_train.drop(corr_features,axis=1,inplace=True)
X_test.drop(corr_features,axis=1,inplace=True)

# standardisation
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title('Before Scaling')
plt.subplot(1,2,2)
sns.boxplot(data=X_train_scaled)
plt.title('After Scaling')
# plt.show()


# Model Training and regression
regression=LinearRegression()
regression.fit(X_train_scaled,Y_train)
Y_pred=regression.predict(X_test_scaled)
mae=mean_absolute_error(Y_test,Y_pred)
mse=mean_squared_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
r2=r2_score(Y_test,Y_pred)
print("Mean Absolute Error:",mae)
print("Mean Squared Error:",mse)        
print("Root Mean Squared Error:",rmse)
print("R2 Score:",r2)
plt.scatter(Y_test,Y_pred)
plt.show()


lasso= Lasso()
lasso.fit(X_train_scaled,Y_train)
Y_pred=lasso.predict(X_test_scaled)
mae=mean_absolute_error(Y_test,Y_pred)
mse=mean_squared_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
r2=r2_score(Y_test,Y_pred)
print("Mean Absolute Error:",mae)
print("Mean Squared Error:",mse)        
print("Root Mean Squared Error:",rmse)
print("R2 Score:",r2)
plt.scatter(Y_test,Y_pred)
plt.show()



ridge= Ridge()
ridge.fit(X_train_scaled,Y_train)
Y_pred=ridge.predict(X_test_scaled)
mae=mean_absolute_error(Y_test,Y_pred)
mse=mean_squared_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
r2=r2_score(Y_test,Y_pred)
print("Mean Absolute Error:",mae)
print("Mean Squared Error:",mse)        
print("Root Mean Squared Error:",rmse)
print("R2 Score:",r2)
plt.scatter(Y_test,Y_pred)
plt.show()

# then same for elasticNet
# so we'll compare all the 3 regression models based on their r2 score and error metrics

## used to check the performance of the model for different alphas
lassocv= LassoCV(cv=5)
lassocv.fit(X_train_scaled,Y_train)
print(lassocv.alpha_)
print(lassocv.alphas_)
## can do such cv for ridge too using RidgeCV


pickle.dump(scaler,open('scaler.pkl','wb'))
pickle.dump(ridge,open('ridge.pkl','wb'))

