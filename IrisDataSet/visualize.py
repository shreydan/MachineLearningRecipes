from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()

test_idx = [0,50,100]

train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data,train_target)

# the above code is same is classify.py but without comments

# TIME TO VISUALIZE THE DecisionTreeClassifier()

# not following PEP8, sorry.

# Also, the following code is from
# scikit-learn, no need to understand
# as for now.

import pydot
from sklearn.externals.six import StringIO
# also requires graphviz - graphviz.com

dot_data = StringIO()
tree.export_graphviz(clf,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True,rounded=True,
                         impurity=False)
                         
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

# unfortunately, it didn't work for me.
# Therefore I've provided iris.pdf from another user
# who followed this tutorial:
# https://github.com/AmruthPillai/MachineLearningRecipes

