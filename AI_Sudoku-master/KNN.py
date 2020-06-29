'''This is a K Nearest Neighbours algorithm with k=3'''
import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle

k = 3
class KNN:

    def __init__(self, k):
        self.mnist = datasets.fetch_openml('mnist_784', data_home='mnist_dataset/')
        self.data, self.target = self.mnist.data, self.mnist.target
        self.indx = np.random.choice(len(self.target), 70000, replace=False)
        self.classifier = KNeighborsClassifier(n_neighbors=k)

    def mk_dataset(self, size):
        
        train_img = [self.data[i] for i in self.indx[:size]]
        train_img = np.array(train_img)
        train_target = [self.target[i] for i in self.indx[:size]]
        train_target = np.array(train_target)

        return train_img, train_target

    def skl_knn(self):
        fifty_x, fifty_y = self.mk_dataset(50000)
        test_img = [self.data[i] for i in self.indx[60000:70000]]
        test_img1 = np.array(test_img)
        test_target = [self.target[i] for i in self.indx[60000:70000]]
        test_target1 = np.array(test_target)
        self.classifier.fit(fifty_x, fifty_y)

        y_pred = self.classifier.predict(test_img1)
        pickle.dump(self.classifier, open('knn.sav', 'wb'))
        print(classification_report(test_target1, y_pred))
        print("KNN Classifier model saved as knn.sav!")
