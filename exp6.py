import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris=load_iris()
iris
X=iris.data
print(X)
Y=iris.target
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
treemodel=DecisionTreeClassifier(max_depth=2)
treemodel.fit(X_train,Y_train)

from sklearn import tree
plt.figure(figsize=(5,5))
tree.plot_tree(treemodel,filled=True)

y_predict=treemodel.predict(X_test)
print(y_predict)

from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(y_predict,Y_test)
print("score",score)
print(classification_report(y_predict,Y_test))
