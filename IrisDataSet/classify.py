from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()

"""
handling the dataset:
    
    iris.feature_names :
        sepal_length
        sepal_width
        petal_length
        petal_width
        
    iris.target_names :
        setosa
        versicolor
        virginica
        
    iris.data[0] :
        prints first item of list feature_names.
            
    iris.target[0] :
        prints 0 which is "setosa" of target_names.
"""


# split training and testing data.

# each element of test_idx list represents
# the starting index of each species.
test_idx = [0,50,100]

# creating training data:

# using numpy, delete elements of test_idx from iris.target list which contains the targets.
train_target = np.delete(iris.target,test_idx)
# using numpy, delete elements of test_idx from iris.data list which contains the actual data.
train_data = np.delete(iris.data,test_idx,axis=0)
# axis = 0 means to remove an entire row.

# creating testing data:

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


# training the classifier:

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data,train_target)

print (test_target) # testing example
print (clf.predict(test_data))

# if test_target matches with clf.predict
# It predicted correctly.

"""
OUTPUT:

[0 1 2]
[0 1 2]

__________

This means that:

The test_target was Setosa, Versicolor 
and Virginica which is [0 1 2].
The prediction was [0 1 2] which says that
this thing predicted all 3 species correctly
which were at element 0,50,100 of iris.data list.


Also,
    instead of randomly selecting test_targets
    like we did in AppleOrange_classifier,
    we separated test_data and train_data from
    same iris data set and used the former to
    test the model and latter to train the model.
"""
