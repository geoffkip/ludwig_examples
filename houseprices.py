from ludwig import LudwigModel
import pandas as pd
import numpy as np
from os import chdir
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Transform features for training
def transform(features):
    categorical =  features.select_dtypes(include=[np.object])
    categorical.fillna('Unknown',inplace=True)
    features_cat = pd.get_dummies(categorical)

    # Numerical Variables
    numerical = features.select_dtypes(include=[np.float64,np.int64])
    numerical.drop(['YearBuilt','YearRemodAdd','YrSold'],axis=1,inplace=True)
    numerical.fillna(method="ffill",inplace=True)

    #Create training features
    train_features= pd.concat([features_cat, numerical], axis=1)
    return train_features

# Add missing columns
def add_missing_dummy_columns( data, columns ):
    missing_cols = set( columns ) - set( data.columns )
    for col in missing_cols:
        data[col] = 0

pd.options.display.max_rows = 999

wd = "/Users/geoffrey.kip/Projects/ludwig/data/"
chdir(wd)

train_df = pd.read_csv("houseprices/train.csv")

#Exploratory analysis
print(train_df.columns)
print(train_df.dtypes)
print(train_df.describe)
print(train_df.isnull().sum())

#Convert types
train_df['MSSubClass']= train_df['MSSubClass'].astype('object')

# Split labels and features
labels = train_df.iloc[:,-1]
features = train_df.loc[:, train_df.columns != 'SalePrice']
features.drop(['Id'],axis=1 ,inplace=True)

#Transform variables
#Create training features
train_features= transform(features)

#Split data into training and test sets
test_size = 0.30
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(train_features, labels, test_size=test_size, random_state=seed)
print (X_train.shape, Y_train.shape)
#print (X_validation.shape, Y_validation.shape)
print (X_test.shape, Y_test.shape)

# Test models
seed = 7
scoring = 'neg_mean_squared_error'
# Evaluate training accuracy
models = []


models.append(('Elastic Net', ElasticNet()))
models.append(('KNN',  KNeighborsRegressor()))
models.append(('Decision Tree', DecisionTreeRegressor()))
models.append(('Support Vector', SVR()))

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "MAE: %s %.3f (%.3f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure(figsize=(10,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

model =  ElasticNet()
model.fit(X_train, Y_train)

feat_imp = pd.Series(model.feature_importances_, X_train.columns).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

best_features= feat_imp[feat_imp > 0]
best_features_columns= list(best_features.index)

predictions = model.predict(X_test)
print(mean_absolute_error(Y_test,predictions))
print(mean_squared_error(Y_test,predictions))
print(r2_score(Y_test,predictions))

prediction_df= pd.DataFrame(predictions, columns=["prediction"])
real_df= pd.DataFrame(Y_test).reset_index(drop=True)
comparison_data= pd.merge(real_df , prediction_df, how='left', left_index=True, right_index=True)
comparison_data["Residual"] = comparison_data["SalePrice"] - comparison_data["prediction"]
comparison_data.head(50)

# Store columns
model_columns=list(X_train.columns)

# Predict on test data
test_df = pd.read_csv("test.csv")
Ids = pd.DataFrame(test_df.loc[:,'Id'],columns=['Id'])
test_features = transform(test_df)

add_missing_dummy_columns(test_features,model_columns)
test_features= test_features[model_columns]

test_prediction= pd.DataFrame(model.predict(test_features),columns=['prediction'])
submission = Ids.merge(test_prediction,how='left',left_index=True, right_index=True)
submission.to_csv('submission.csv')

# Try with ludwig
# train a model
#load a model
model = LudwigModel.load("/Users/geoffrey.kip/Projects/ludwig/ludwig_models/results/experiment_run_1/model")

# obtain predictions
ludwig_predictions = model.predict(test_df)

#evaluate predictions
preds = np.where(predictions_probs[:,1] >= 0.5 , 1, 0)
print(accuracy_score(Y_test, preds))
print(confusion_matrix(Y_test, preds))
print(classification_report(Y_test, preds))

model.close()
