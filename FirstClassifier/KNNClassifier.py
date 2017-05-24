"""
Randomly predict the point of differentiation
find the nearest neighbor to the point.
consider the point of differentiation have same feature as its nearest neighbor.
if 2 neighbors are equidistant, use more neighbors.
which is K - number of neighbors to consider.
vote the no. of nearest neighbors.

to find the straight line distance, we use Euclidean Distance formula
distance = sqrt((x2-x1)^2 + (y2-y1)^2) for 2D
this formula is true for multiple dimensions too:
distance = sqrt((x2-x1)^2 + (y2-y1)^2 ... + (n2-n1)^2)

then we find the distance of first nearest neighbor.
Then we'll loop over the data element by element,
and keep updating the best_distance which returns the target data
The target data is the new label.
"""

from scipy.spatial import distance

def EuclideanDistance(x,y):
    return distance.euclidean(x,y)

class KNNClassifier():
    
    def fit(self,x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    
    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)         
            
        return predictions
    
        
    def closest(self,row):
        best_dist = EuclideanDistance(row,self.x_train[0])
        best_index = 0
        for i in range(1,len(self.x_train)):
            dist = EuclideanDistance(row,self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i 
        
        return self.y_train[best_index]
