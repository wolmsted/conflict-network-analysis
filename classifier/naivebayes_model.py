import os
import sys
import scipy
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def make_data():
	train_examples = np.genfromtxt('train.csv', delimiter=',')
	train_labels = train_examples[:,0]
	train_examples = train_examples[:,1:]
	test_set = np.genfromtxt('test.csv', delimiter=',')
	test_labels = test_set[:,0]
	test_set = test_set[:,1:]

	return train_examples, train_labels, test_set, test_labels


def main():
	train_set, train_y, test_set, test_y = make_data()
	gnb = GaussianNB(tol=0.1)
	y_pred = gnb.fit(train_set, train_y).predict(test_set)
	num_samples = test_y.shape[0]
	num_incorrect = (test_y != y_pred)
	num_correct = (test_y == y_pred)

	accuracy = num_correct.sum() / float(num_samples)
	
	print("Number of mislabeled points out of a total %d points : %d"
		% (num_samples, num_incorrect.sum()))
	print("Test accuracy: ", accuracy)


if __name__ == '__main__':
	main()


