# coding: utf-8

import cPickle, json, pdb, pickle, theano, sys
import os.path
import theano.tensor as T
from dataset.Nikkei import Nikkei
from experiment.CompressSparseVector.SparseAutoencoder import SparseAutoencoder, train_sae
from experiment.CompressSparseVector.RBM import RBM, train_rbm
from experiment.PredictPrices.SdA_theano import SdA, train_SdA
import run
theano.config.floatX = 'float32'

default_model_dir = '/home/fujikawa/StockPredict/src/deeplearning/experiment/Model'

run.run_step234(n_hidden=10000, learning_rate=0.05)