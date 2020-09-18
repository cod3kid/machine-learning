import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB()
clf.fit(X, Y) 
pred=clf.predict([[-1.8, -1]])
print(pred)

# to know how accurate our model is. we can use accuracy_score.
# accuracy_score(pred,test_label)
