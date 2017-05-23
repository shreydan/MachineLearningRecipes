from sklearn import tree

# features = [[mass,texture=[1:smooth,0:bumpy]]]
features = [ 
    [140,1],
    [130,1],
    [150,0],
    [170,0],
    [200,0],
    [100,1],
    [130,0],
    [120,1]
]

#labels -> Maps to each element of features. label[0] => features[0] 
labels = ["Apple","Apple","Orange","Orange","Orange","Apple","Orange","Apple"]

# Training Classifier
#    - The classifier is an empty box of rules and the algorithm creates the rules.
#    - Use Decision Tree Classifier to classify features.
classifier = tree.DecisionTreeClassifier()
#    - Using FIT (find patterns in Data) algorithm to find patterns
classifier = classifier.fit(features,labels)


#predict
print (classifier.predict([160,0]))
