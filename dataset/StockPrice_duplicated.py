import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import cPickle, gzip
datasetdir = "/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/FeatureVectors/chi2-1000/category/0101"

class StockPrice(DenseDesignMatrix):
    def __init__(self, datasetdir=datasetdir):
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