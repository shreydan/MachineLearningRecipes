# KNearestNeighbors
# Going to use code from #4: Pipeline

from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5)

# from sklearn.neighbors import KNeighborsClassifier
# NOW LET'S use the one I created!!!!!!

from KNNClassifier import KNNClassifier
clf = KNNClassifier()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions)*100.0)
