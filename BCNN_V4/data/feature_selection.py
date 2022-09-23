# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:37:12 2021
feature selection
@author: wwang
"""

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
iris = load_iris()
X, y = iris.data, iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


"""网格搜素参数选择"""
def adjust_params(X,y):
    xgb_cls=XGBClassifier(random_state=1, n_jobs=-1, objective='binary:logistic',eval_metric='auc') #,**other_params
     #:error: error rate of binary classifier/auc;  :objective='binary:logistic'
    params_set={'n_estimators':[5,10,15,20],"learning_rate":[0.1,0.001,0.01],
               'max_depth':[i for i in range(3,11)], # depth of tree
               }
    
    grid=GridSearchCV(xgb_cls,cv=5,param_grid=params_set)
    grid.fit(X,y)
    other_params=grid.best_params_
    print("best params:",other_params)
    return other_params

other_params=adjust_params(X_train,y_train)
xgb_cls=XGBClassifier(random_state=1, n_jobs=-1, objective='binary:logistic',eval_metric='auc',**other_params) 
xgb_cls.fit(X_train,y_train)
y_pred=xgb_cls.predict(X_test)
pred=[round(value) for value in y_pred]
acc=accuracy_score(y_test,pred)
print("accuracy: %.2f%%"%(acc*100.0))


"""特征重要度评估,五种计算方式得到特征重要度得分"""

from xgboost import plot_importance
import matplotlib.pyplot as plt
Available_importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover'] 
for importance_type in Available_importance_types:
    print('%s:'% importance_type,xgb_cls.get_booster().get_score(importance_type=importance_type))
    plot_importance(xgb_cls)
    plt.show()
#importance_type='weight'
#xgb_cls.get_booster().get_score(importance_type=importance_type)
#plot_importance(xgb_cls)
#plt.show()

#xgb.to_graphviz(, num_trees=0)




"""递归特征消除-交叉验证"""
from sklearn.feature_selection import RFE, RFECV
# RFE-cv using cross validation to select best features combination
from sklearn.svm import LinearSVC
from sklearn import model_selection

xgb_cls=XGBClassifier(random_state=1, n_jobs=-1, objective='binary:logistic',eval_metric='auc',**other_params) 
estimator = xgb_cls

# using rfecv to select feature
selector=RFECV(estimator=estimator,cv=3)
selector.fit(X_train,y_train)
print("number of features %s"% selector.n_features_)
print("support is %s"% selector.support_)
print("ranking of features is%s" % selector.ranking_)
print("grid scores %s" % selector.grid_scores_)



