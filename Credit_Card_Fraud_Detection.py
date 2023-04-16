import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score 
import tensorflow
from tensorflow.keras.layers import Dense
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')


train_data = pd.DataFrame(pd.read_csv('training_data.csv'))
train_class_labels = pd.DataFrame(pd.read_csv('train_data_classlabels.csv'))
testing_data = pd.DataFrame(pd.read_csv('testing_data.csv'))
testing_data = testing_data.dropna()


N = train_data.shape[0]
print("Number Of transactions in dataset : {}\n".format(N))


print(train_data.head(),"\n")
print(train_class_labels.head(),"\n")
print(testing_data.head(),"\n")



colors = {0.0 : 'blue' , 1.0 : 'orange'}
plt.scatter(train_data['Amount'],train_class_labels['Class'],c=train_class_labels['Class'].map(colors))
plt.show()

frauds_count = (train_class_labels['Class'] == 1.0).sum()
safe_count = train_data.shape[0] - frauds_count
print("Number of frauds : {}".format(frauds_count))
print("Number of safe transactions : {}\n".format(safe_count))



fraud = train_data[train_class_labels['Class'] == 1]
fraud_y = train_class_labels[train_class_labels['Class'] == 1]
safe = train_data[train_class_labels['Class'] == 0]
safe_y = train_class_labels[train_class_labels['Class'] == 0]

safe = safe.iloc[0:42000,:]
safe_y = safe_y.iloc[0:42000,:]

while len(fraud) < 10000 :
  fraud = fraud.append(fraud,ignore_index = True)
  fraud_y = fraud_y.append(fraud_y,ignore_index = True)


train_data = safe.append(fraud,ignore_index = True)
train_class_labels = safe_y.append(fraud_y,ignore_index = True)



X1 = train_data.iloc[:,0:10]
X2 = train_data.iloc[:,10:20]
X3 = train_data.iloc[:,20:30]

X1.insert(10,'Class',train_class_labels,True)
X2.insert(10,'Class',train_class_labels,True)
X3.insert(10,'Class',train_class_labels,True)

fig,axes = plt.subplots(3,figsize=(25,25))

corr1 = X1.corr()
sns.heatmap(corr1,annot=True,ax=axes[0])

corr2 = X2.corr()
sns.heatmap(corr2,annot=True,ax=axes[1])

corr3 = X3.corr()
sns.heatmap(corr3,annot=True,ax=axes[2])
plt.show()


model = ExtraTreesClassifier()
model.fit(train_data,train_class_labels)
importances = model.feature_importances_
importance_normalised = np.std([tree.feature_importances_ for tree in model.estimators_],axis = 0)

plt.figure(figsize=(16,10))
plt.bar(train_data.columns, importance_normalised)
plt.xlabel('Features')
plt.ylabel('Feature Importances')
plt.title('Comparison of different Feature Importances')
plt.show()



def LogisticRegressionModel() :

  logistic_model = LogisticRegression()            # Creating Model 

  grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"], "solver" :['liblinear','newton-cg','saga','sag']}

  logis_model = GridSearchCV(logistic_model, grid, cv=10)

  logis_model.fit(train_data,train_class_labels)

  preds = logis_model.predict(train_data)

  print("Tuned Paramters (Best Hyperparameters) : ",logis_model.best_params_)
  print("Accuracy Of The Model : {}%".format(logis_model.best_score_*100))
  print("Best Estimator Settings : ",logis_model.best_estimator_)
  print("\n")
  print("Classification Report\n")
  print(classification_report(train_class_labels, preds))
  print("\n")

  print("************** Test Data *************\n")

  preds2 = logis_model.predict(testing_data)
  print('Number Of Frauds : ',preds2.sum())
  print('Safe Transactions : ', len(preds2) - preds2.sum())





def RandomForestModel() :

  rand_forest_model = RandomForestClassifier()

  grid = {'criterion' : ['gini','entropy'],'max_depth': list(range(1,10))}

  random_for_model = GridSearchCV(rand_forest_model, grid, cv = 10, scoring = 'f1')

  random_for_model.fit(train_data,train_class_labels)

  preds = random_for_model.predict(train_data)

  print("Tuned Paramters (Best Hyperparameters) : ",random_for_model.best_params_)
  print("Accuracy Of The Model : {}%".format(random_for_model.best_score_*100))
  print("Best Estimator Settings : ",random_for_model.best_estimator_)
  print("\n")
  print("Classification Report\n")
  print(classification_report(train_class_labels, preds))
  print("\n")

  print("************** Test Data *************\n")

  preds2 = random_for_model.predict(testing_data)
  print('Number Of Frauds : ',preds2.sum())
  print('Safe Transactions : ', len(preds2) - preds2.sum())





def KNearestNeighborsModel() :

  KNN_model = KNeighborsClassifier()

  grid = {'n_neighbors' : list(range(3,50)),'metric' : ['minkowski','euclidean']}

  KNNei_model = GridSearchCV(KNN_model, grid, cv = 10, scoring='f1', return_train_score=False)

  KNNei_model.fit(train_data,train_class_labels)

  preds = KNNei_model.predict(train_data)

  print("Tuned Paramters (Best Hyperparameters) : ",KNNei_model.best_params_)
  print("Accuracy Of The Model : {}%".format(KNNei_model.best_score_*100))
  print("Best Estimator Settings : ",KNNei_model.best_estimator_)
  print("\n")
  print("Classification Report\n")
  print(classification_report(train_class_labels, preds))
  print("\n")

  print("************** Test Data *************\n")

  preds2 = KNNei_model.predict(testing_data)
  print('Number Of Frauds : ',preds2.sum())
  print('Safe Transactions : ', len(preds2) - preds2.sum())






def build_model(optimizer,loss):
  model = tensorflow.keras.Sequential([
                  Dense(units = 40,input_shape = [train_data.shape[1]],activation = 'relu'),
                  Dense(units = 30, activation = 'relu'),
                  Dense(units = 20, activation = 'relu'),
                  Dense(units = 10, activation = 'relu'),
                  Dense(units = 1, activation = 'sigmoid'),
                  ])
  model.compile(loss=loss,optimizer=optimizer)
  return model

def NeuralNetworkModel():

  neural_model = KerasClassifier(build_fn=build_model)

  grid = {'batch_size':[100,500],'epochs':[20,30,50],'optimizer': ['adam'],'loss' : ['binary_crossentropy']}

  neur_model = GridSearchCV(estimator = neural_model, param_grid = grid,cv = 10, scoring = 'f1')

  neur_model.fit(train_data,train_class_labels)

  preds = neur_model.predict(train_data)

  print("Tuned Paramters (Best Hyperparameters) : ",neur_model.best_params_)
  print("Accuracy Of The Model : {}%".format(neur_model.best_score_*100))
  print("Best Estimator Settings : ",neur_model.best_estimator_)
  print("\n")
  print("Classification Report\n")
  print(classification_report(train_class_labels, preds))
  print("\n")

  print("************** Test Data *************\n")







def SupportVectorMachine():
  svm_model = svm.SVC()
  grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','poly','sigmoid']} 
  svm_model1 = GridSearchCV(estimator = svm_model, param_grid=grid, cv=10, scoring = 'f1')
  svm_model1.fit(train_data,train_class_labels)
  preds= svm_model1.predict(train_data)
  print("Tuned Parameters (Best  Hyperparameters) : ",svm_model1.best_params_)
  print("F Score of the model : {}".format(svm_model1.best_score_*100))
  print("Best Estimator Settings : ",svm_model1.best_estimator_)
  print("\n")
  print("Classifiaction Report\n")
  print(classification_report(train_class_labels,preds))
  print("\n")

  print("************Test Data***************\n")
  preds2 = svm_model1.predict(testing_data)
  print('Number Of Frauds : ',preds2.sum())
  print('Safe Transactions : ', len(preds2) - preds2.sum())


def AdaBoostModel():
  abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

  parameters = {'base_estimator__max_depth':[i for i in range(1,5,1)],
              'base_estimator__min_samples_leaf':[5,10],
              'n_estimators':[10,50,250],
              'learning_rate':[0.1]}

  ada_boost_model = GridSearchCV(abc, parameters,verbose=3,scoring='f1',n_jobs=-1)
  ada_boost_model.fit(train_data,train_class_labels)
  preds= ada_boost_model.predict(train_data)
  print("Tuned Parameters (Best  Hyperparameters) : ",ada_boost_model.best_params_)
  print("F Score of the model : {}".format(ada_boost_model.best_score_*100))
  print("Best Estimator Settings : ",ada_boost_model.best_estimator_)
  print("\n")
  print("Classifiaction Report\n")
  print(classification_report(train_class_labels,preds))
  print("\n")

  print("************Test Data***************\n")
  preds2 = ada_boost_model.predict(testing_data)
  print('Number Of Frauds : ',preds2.sum())
  print('Safe Transactions : ', len(preds2) - preds2.sum())




def DecisionTreeModel():
  grid = {'criterion': ['gini','entropy'],'max_leaf_nodes': list(range(2, 10)), 'min_samples_split': [2, 3, 4]}
  decision_model = GridSearchCV(DecisionTreeClassifier(random_state=42),grid,verbose=1,cv = 10)

  decision_model.fit(train_data,train_class_labels)
  preds= decision_model.predict(train_data)
  print("Tuned Parameters (Best  Hyperparameters) : ",decision_model.best_params_)
  print("F Score of the model : {}".format(decision_model.best_score_*100))
  print("Best Estimator Settings : ",decision_model.best_estimator_)
  print("\n")
  print("Classifiaction Report\n")
  print(classification_report(train_class_labels,preds))
  print("\n")

  print("************Test Data***************\n")
  preds2 = decision_model.predict(testing_data)
  print('Number Of Frauds : ',preds2.sum())
  print('Safe Transactions : ', len(preds2) - preds2.sum())



LogisticRegressionModel()
RandomForestModel()
KNearestNeighborsModel()
NeuralNetworkModel()
SupportVectorMachine()
AdaBoostModel()
DecisionTreeModel()
