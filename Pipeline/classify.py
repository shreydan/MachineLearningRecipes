from sklearn import datasets
iris = datasets.load_iris()

x = iris.data #features
y= iris.target #labels

# f(x) = y 
#classifier is a function which takes your features and returns a label

"""
basic classifier:

def classifierName(features):
    // do some stuff.
    return label
"""

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5)
# this is much better than numpy.
#test_size = .5 denotes 50% for each set train and test.


# USING DecisionTreeClassifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
predictions_DTC = clf.predict(x_test)


# USING KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf = clf.fit(x_train,y_train)
predictions_KNC = clf.predict(x_test)


#check accuracy of predictions
from sklearn.metrics import accuracy_score
print ("Using DTC {}".format(accuracy_score(y_test, predictions_DTC)*100.0))
print ("Using KNC {}".format(accuracy_score(y_test, predictions_KNC)*100.0))
