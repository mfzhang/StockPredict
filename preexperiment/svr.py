from StockPrice import StockPrice
from sklearn.svm import SVR
dataset = StockPrice()

import numpy as np
print dataset.train[1]

class Accuracy(object):
	def __init__(self):
		self.p = 0
		self.n = 0
	def evaluatePN(self, predict, label):
		if (predict > 0 and label > 0) or (predict <= 0 and label <= 0):
			self.p += 1
		else:
			self.n += 1
	def printResult(self):
		print "p: %s, n: %s, rate: %s" % (str(self.p), str(self.n), str(float(self.p) / (self.p + self.n)))


def transformY(data_y):
	y = []
	for data in data_y:
		y.append(data[0])
	return np.array(y)

# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()
# print y
train_X = dataset.train[0]
train_Y = transformY(dataset.train[1])
test_X = dataset.valid[0]
test_Y = transformY(dataset.valid[1])


svr = SVR(C=1e3, gamma=0.1)
y = svr.fit(train_X, train_Y).predict(test_X)

acc = Accuracy()
for i, predict in enumerate(y):
	print predict, dataset.valid[1][i]
	acc.evaluatePN(predict, dataset.valid[1][i])
acc.printResult()