# coding: utf-8
import cPickle, json, pdb, pickle, theano, sys, time, os, codecs
import numpy as np
from progressbar import ProgressBar
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
import os.path
import copy
import theano.tensor as T
from dataset.Nikkei import Nikkei
from experiment.CompressSparseVector.SparseAutoencoder import SparseAutoencoder, train_sae
from experiment.CompressSparseVector.RBM import RBM, train_rbm

from experiment.PredictPrices import SdA_regression
from experiment.PredictPrices import SdA_RNN
from experiment.PredictPrices import DBN_regression

from experiment.PredictPrices.RNN import train_RNN, train_RNN_hf, train_RNN_minibatch
import curses
from run import *


def test_phase1(params, model_dirs):
    model = load_model(input=x, params_dir=model_dirs['STEP1'], model_type=params['model'])
    dic = json.load(codecs.open('dataset/dataset/chi2-result-unified.rdic', 'r', 'utf-8'))
    threshold = 0.2
    max_len = 10
    out = codecs.open(model_dirs['STEP1_analysis'], 'w', 'utf-8')
    bar = ProgressBar(maxval=model.W.get_value().shape[0]).start()
    for i in range(model.W.get_value().shape[0]):
        bar.update(i)
        best_args = model.W.get_value()[i].argsort()[::-1]
        W_values = []

        for arg in best_args:
            if model.W.get_value()[i][arg] > threshold:

                W_values.append(dic[str(arg)])
                W_values.append(str(model.W.get_value()[i][arg]))
                if len(W_values) > max_len * 2:
                    break
            else:
                break
        # pdb.set_trace()
        # print i
        out.write(','.join(W_values) + '\n')
    out.close()



if __name__ == '__main__':
    params = {}
    if len(sys.argv) > 5:
        params['n_hidden'] = int(sys.argv[1])
        params['learning_rate'] = float(sys.argv[2])
        params['reg_weight'] = float(sys.argv[3])
        params['corruption_level'] = float(sys.argv[4])
        params['model'] = sys.argv[5]
    else:
        print sys.argv
        print '引数が足りません．'
        print '引数: n_hidden learning_rate reg_weight model'
        sys.exit()
    default_model_dir_prefix = 'experiment/Model/'
    default_model_dir_suffix = '/h%d_lr%s_b%s_c%s.%s' % (params['n_hidden'], str(params['learning_rate']), str(params['reg_weight']), str(params['corruption_level']), params['model'])
    model_dirs = {
        'STEP1' : default_model_dir_prefix + 'STEP1' + default_model_dir_suffix,
        'STEP1_analysis' : default_model_dir_prefix + 'STEP1_analysis' + default_model_dir_suffix,
        'STEP2' : default_model_dir_prefix + 'STEP2' + default_model_dir_suffix,
        'STEP3' : default_model_dir_prefix + 'STEP3' + default_model_dir_suffix
    }
    test_phase1(params, model_dirs)
