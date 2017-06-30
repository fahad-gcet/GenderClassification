from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.metrics import accuracy_score

# Train data
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47],
	 [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Test data
X_test=[[198,92,48], [184,84,44], [183,83,44], [166,47,36], [170,60,38], [172,64,39],
		[182,80,42], [180,80,43]]
y_test=['male', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Decision Tree Classifier
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(X, y)
y_prediction1 = clf1.predict(X_test)

# Support Vector Classifier
clf2 = svm.SVC()
clf2 = clf2.fit(X, y)
y_prediction2 = clf2.predict(X_test)

# Gaussian Naive Bayes Classifier
clf3 = GaussianNB()
clf3 = clf3.fit(X, y)
y_prediction3 = clf3.predict(X_test)

# K Neighbors Classifier
clf4 = neighbors.KNeighborsClassifier()
clf4 = clf4.fit(X, y)
y_prediction4 = clf4.predict(X_test)

print("Decision Tree Classifier Accuracy: ", accuracy_score(y_test, y_prediction1))
print("Support Vector Classifier Accuracy: ", accuracy_score(y_test, y_prediction2))
print("Gaussian Naive Bayes Classifier Accuracy: ", accuracy_score(y_test, y_prediction3))
print("K Neighbors Classifier Accuracy: ", accuracy_score(y_test, y_prediction4))