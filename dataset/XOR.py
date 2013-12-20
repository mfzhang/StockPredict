# coding: utf-8
import sys, theano
sys.path.append('/home/fujikawa/lib/python/other/pylearn2/pylearn2')
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.sparse_dataset import SparseDataset
import numpy as np
from random import randint
import scipy.sparse
import theano
floatX = theano.config.floatX

class XOR(DenseDesignMatrix):
    def __init__(self, size=1000, type='rnd'):
        # type = 'rnd' : create random data
        #      = 'seq' : create sequence data
        self.class_names = ['0', '1']
        self.size = size
        if type == 'rnd':
            X = [[randint(0, 1), randint(0, 1)] for _ in range(size)]
        else:
            lists = [[0, 0], [1, 1]]
            X = [lists[i % 2] for i in range(size)]
            
        y = []
        for a, b in X:
            if a + b == 1:
                y.append([0, 1])
            else:
                y.append([1, 0])
        self.X = np.array(X)
        self.y = np.array(y)
        super(XOR, self).__init__(X=self.X, y=self.y)
    def get_train_data(self):
        return self.get_theano_sparse_design()
    def get_theano_design(self):
        return theano.shared(np.asarray(self.X, dtype=theano.config.floatX))
    def get_theano_sparse_design(self):
        return theano.shared(scipy.sparse.csr_matrix(np.asarray(self.X, dtype=theano.config.floatX)))
    def get_scipy_sparse_design(self):
        return scipy.sparse.csr_matrix(self.X)
    def get_batch_design(self, batch_size):
        n_batches = self.size / batch_size
        batch_design_data = []
        for i in range(n_batches):
            batch_design_data.append(self.X[batch_size * i: batch_size * (i + 1)])
        return batch_design_data        

    def test(self, model):
        print '#######    TEST    #######'
        print 'Input: [0, 0] -> Predict: ' + str(model.fprop(theano.shared(np.array([[0, 0]]), name='inputs')).eval()) + ' (Correct: [1, 0])'
        print 'Input: [0, 1] -> Predict: ' + str(model.fprop(theano.shared(np.array([[0, 1]]), name='inputs')).eval()) + ' (Correct: [0, 1])'
        print 'Input: [1, 0] -> Predict: ' + str(model.fprop(theano.shared(np.array([[1, 0]]), name='inputs')).eval()) + ' (Correct: [0, 1])'
        print 'Input: [1, 1] -> Predict: ' + str(model.fprop(theano.shared(np.array([[1, 1]]), name='inputs')).eval()) + ' (Correct: [1, 0])'
        
if __name__ == '__main__':
    data = XOR()
    print data.get_theano_sparse_design()


