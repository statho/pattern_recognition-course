import numpy as np
from sklearn import svm, preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier

x_train=[]
with open('train.txt') as ftrain:
	for line in ftrain:
		x_train.append(map(float, line.split()))

x_test=[]
with open('train.txt') as ftest:
	for line in ftest:
		x_test.append(map(float, line.split()))	


Y_train = np.array([i[0] for i in x_train])
X_train =  np.array([i[1:] for i in x_train])

Y_test = np.array([i[0] for i in x_test])
X_test = np.array([i[1:] for i in x_test])

# Normalize Traing and Test Data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test)

expected = Y_test

# SVM using Linear Kernel
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train, Y_train)

predicted = svm_linear.predict(X_test)
print "Classification report for classifier %s:\n%s\n" % (svm_linear, metrics.classification_report(expected, predicted))
print "Accuracy of SVM_LINEAR :{}".format(metrics.accuracy_score(expected, predicted))

# SVM using 3 Degree Polynomial Kernel
svm_poly = svm.SVC(kernel='poly', degree=3)
svm_poly.fit(X_train, Y_train)

predicted = svm_poly.predict(X_test)
print "Classification report for classifier %s:\n%s\n" % (svm_poly, metrics.classification_report(expected, predicted))
print "Accuracy of SVM_POLY : {}".format(metrics.accuracy_score(expected, predicted))


# Random Forests Classifier
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, Y_train)
predicted = clf_rf.predict(X_test)
print "Classification report for classifier %s:\n%s\n" % (clf_rf, metrics.classification_report(expected, predicted))
print "Accuracy of RANDOM_FOREST : {}".format(metrics.accuracy_score(expected, predicted))