import theano, sys, cPickle, gzip
sys.path.append('/home/fujikawa/lib/python/other/pylearn2/pylearn2')
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.sparse_dataset import SparseDataset
import numpy as np
from random import randint
# autoencoder
import theano.tensor as tensor
from theano import config
from pylearn2.models.autoencoder import DenoisingAutoencoder, HigherOrderContractiveAutoencoder
from pylearn2.corruption import BinomialCorruptor
from theano.tensor.basic import _allclose
import pylearn2.costs.autoencoder as cost_ae

import scipy.sparse as sp
datasetdir = "/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/FeatureVectors/chi2-1000/category/0101"



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

class StockPrice(DenseDesignMatrix):
    def __init__(self):
        dataset = cPickle.load(gzip.open(datasetdir))
        self.train = self.getnparrays([self.idlists2VectorData(dataset[0][0]), dataset[0][1]])
        self.valid = self.getnparrays([self.idlists2VectorData(dataset[1][0]), dataset[1][1]])
        self.test = self.getnparrays([self.idlists2VectorData(dataset[2][0]), dataset[2][1]])
        super(StockPrice, self).__init__(X=self.train[0], y=self.train[1])
    def idlists2VectorData(self, idlists):
        vectors = []
        for idlist in idlists:
            vector = [0 for i in range(1000)]
            for id in idlist:
                vector[int(id)] = 1
            vectors.append(vector)
        return vectors
    def getnparrays(self, xy):
        x, y = xy
        return [np.array(x), np.array(y)]

class SparseStockPrice(SparseDataset):
    def __init__(self):
        dataset = cPickle.load(gzip.open(datasetdir))
        data = sp.eye(200, 20, k=0, format='csr')
        print data.todense()
        # self.train = self.getsparrays([self.idlists2VectorData(dataset[0][0]), dataset[0][1]])
        # self.valid = self.getsparrays([self.idlists2VectorData(dataset[1][0]), dataset[1][1]])
        # self.test = self.getsparrays([self.idlists2VectorData(dataset[2][0]), dataset[2][1]])
        self.train = self.idlist2SparseVectorData(dataset[0][0])
        super(SparseStockPrice, self).__init__(from_scipy_sparse_dataset=self.train)
    def idlists2VectorData(self, idlists):
        vectors = []
        for idlist in idlists:
            vector = [0 for i in range(1000)]
            for id in idlist:
                vector[int(id)] = 1
            vectors.append(vector)
        return vectors
    def idlist2SparseVectorData(self, idlists):
        data = []
        row = []
        col = []
        for i, idlist in enumerate(idlists):
            for id in idlist:
                row.append(i)
                col.append(id)
                data.append(1)
        return sp.csc_matrix((data, (row, col)), shape=(len(idlists), 1000))
        # return [sp.csc_matrix(x), sp.csc_matrix(y)]
    def getnparrays(self, xy):
        x, y = xy
        return [np.array(x), np.array(y)]


def runAutoencoder():
    ds = StockPrice()
    #print ds.train[0][0]
    data = np.random.randn(10, 5).astype(config.floatX)
    #print data
    print BinomialCorruptor(.2)
    ae = DenoisingAutoencoder(BinomialCorruptor(corruption_level=.2), 1000, 100, act_enc='sigmoid', act_dec='linear',
                     tied_weights=False)
    trainer = sgd.SGD(learning_rate=.005, batch_size=5, termination_criterion=EpochCounter(3), cost=cost_ae.MeanSquaredReconstructionError(), monitoring_batches=5, monitoring_dataset=ds)
    trainer.setup(ae, ds)
    while True:
        trainer.train(dataset=ds)
        ae.monitor()
        ae.monitor.report_epoch()
        if not trainer.continue_learning(ae):
            break
    #print ds.train[0][0]
    #print ae.reconstruct(ds.train[0][0])

    w = ae.weights.get_value()
    #ae.hidbias.set_value(np.random.randn(1000).astype(config.floatX))
    hb = ae.hidbias.get_value()
    #ae.visbias.set_value(np.random.randn(100).astype(config.floatX))
    vb = ae.visbias.get_value()
    d = tensor.matrix()
    result = np.dot(1. / (1 + np.exp(-hb - np.dot(ds.train[0][0],  w))), w.T) + vb
    #print result

def runSP():
    ds = StockPrice()
    
    # create hidden layer with 2 nodes, init weights in range -0.1 to 0.1 and add
    # a bias with value 1
    hidden_layer = mlp.Sigmoid(layer_name='hidden', dim=10000, irange=.1, init_bias=1.)
    # create Softmax output layer
    output_layer = mlp.Linear(layer_name='output', dim=1, irange=.1, init_bias=1.)
    # create Stochastic Gradient Descent trainer that runs for 400 epochs
    trainer = sgd.SGD(learning_rate=.005, batch_size=500, termination_criterion=EpochCounter(10))
    layers = [hidden_layer, output_layer]
    # create neural net that takes two inputs
    ann = mlp.MLP(layers, nvis=1000)
    trainer.setup(ann, ds)
    # train neural net until the termination criterion is true
    while True:
        trainer.train(dataset=ds)
        ann.monitor.report_epoch()
        ann.monitor()
        if not trainer.continue_learning(ann):
            break
    #accuracy = Accuracy()
    acc = Accuracy()
    for i, predict in enumerate(ann.fprop(theano.shared(ds.valid[0], name='inputs')).eval()):
        print predict, ds.valid[1][i]
        acc.evaluatePN(predict[0], ds.valid[1][i][0])
    acc.printResult()
    

 
class XOR(DenseDesignMatrix):
    def __init__(self):
        #self.class_names = ['0', '1']
        X = [[randint(0, 1), randint(0, 1), randint(0, 1), randint(0, 1)] for _ in range(1000)]
        y = []
        for a, b, c, d in X:
            if (a + b + c + d) % 2 == 0:
                y.append([1.5])
            else:
                y.append([0.1])
        X = np.array(X)
        y = np.array(y)
        super(XOR, self).__init__(X=X, y=y)

def runXOR():
    ds = XOR()
    hidden_layer = mlp.Sigmoid(layer_name='hidden', dim=10, irange=.1, init_bias=1.)
    output_layer = mlp.Linear(layer_name='output', dim=1, irange=.1, init_bias=1.)
    trainer = sgd.SGD(learning_rate=.05, batch_size=1, termination_criterion=EpochCounter(1000))
    layers = [hidden_layer, output_layer]
    # create neural net that takes two inputs
    ann = mlp.MLP(layers, nvis=4)
    trainer.setup(ann, ds)
    # train neural net until the termination criterion is true
    while True:
        trainer.train(dataset=ds)
        #ann.monitor.report_epoch()
        #ann.monitor()
        if not trainer.continue_learning(ann):
            break
    inputs= np.array([[0, 0, 0, 1]])
    print ann.fprop(theano.shared(inputs, name='inputs')).eval()
    inputs = np.array([[0, 1, 0, 1]])
    print ann.fprop(theano.shared(inputs, name='inputs')).eval()
    inputs = np.array([[1, 1, 1, 1]])
    print ann.fprop(theano.shared(inputs, name='inputs')).eval()
    inputs = np.array([[1, 1, 0, 0]])
    print ann.fprop(theano.shared(inputs, name='inputs')).eval()

if __name__ == '__main__':
    runAutoencoder()

#runSP()
 
