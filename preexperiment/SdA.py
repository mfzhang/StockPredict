import theano, sys, cPickle, gzip
sys.path.append('/home/fujikawa/lib/python/other/pylearn2/pylearn2')
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.train import Train
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.sparse_dataset import SparseDataset
from pylearn2.datasets.transformer_dataset import TransformerDataset
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

def runSdA():

    ds = StockPrice()
    
    ########################
    ####  Pre training  ####
    ########################
    ### First layer ###
    dA_1 = DenoisingAutoencoder(BinomialCorruptor(corruption_level=.2), 1000, 500, act_enc='sigmoid', act_dec='linear',
                     tied_weights=False)
    train_1 = Train(
        ds,
        dA_1,
        algorithm=sgd.SGD(learning_rate=1e-3, batch_size=100, termination_criterion=EpochCounter(10), cost=cost_ae.MeanSquaredReconstructionError(), monitoring_batches=5, monitoring_dataset=ds)
    )
    train_1.main_loop()

    dA_1_out = TransformerDataset(ds, dA_1)

    ### Second layer ###
    dA_2 = DenoisingAutoencoder(BinomialCorruptor(corruption_level=.3), 500, 500, act_enc='sigmoid', act_dec='linear',
                     tied_weights=False)
    train_2 = Train(
        dA_1_out,
        dA_2,
        algorithm=sgd.SGD(learning_rate=1e-3, batch_size=100, termination_criterion=EpochCounter(10), cost=cost_ae.MeanSquaredReconstructionError(), monitoring_batches=5, monitoring_dataset=dA_1_out)
    )
    train_2.main_loop()
    
    #######################
    ####  Fine tuning  ####
    #######################
    
    ### defining each layers ###
    layer_1 = mlp.PretrainedLayer('layer_1', dA_1)
    layer_2 = mlp.PretrainedLayer('layer_2', dA_2)
    output_layer = mlp.Linear(layer_name='output', dim=1, irange=.1, init_bias=1.)

    ### run fine tuning ###
    layers = [layer_1, layer_2, output_layer]
    main_mlp = mlp.MLP(layers, nvis=1000)
    train = Train(
        ds,
        main_mlp,
        algorithm=sgd.SGD(learning_rate=.05, batch_size=4, termination_criterion=EpochCounter(50), monitoring_batches=4, monitoring_dataset=ds)
    )
    train.main_loop()

    acc = Accuracy()
    for i, predict in enumerate(main_mlp.fprop(theano.shared(ds.valid[0], name='inputs')).eval()):
        print predict, ds.valid[1][i]
        acc.evaluatePN(predict[0], ds.valid[1][i][0])
    acc.printResult()
    

if __name__ == '__main__':
    runSdA()

#runSP()
 
