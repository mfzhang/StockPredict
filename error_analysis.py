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

from experiment.PredictPrices import SdA
from experiment.PredictPrices import SdA_RNN
from experiment.PredictPrices import DBN

from experiment.PredictPrices.RNN import train_RNN, train_RNN_hf, train_RNN_minibatch
import curses
from run import *


def test_phase1(params, model_dirs):
    model = load_model(input=x, params_dir=model_dirs['STEP1'], model_type=params['model'])
    dic = json.load(codecs.open('dataset/dataset/chi2-result-unified_10000.rdic', 'r', 'utf-8'))
    threshold = 0.2
    max_len = 10

    out = codecs.open(model_dirs['STEP1_analysis'], 'w', 'shift-jis')
    # bar = ProgressBar(maxval=model.W.get_value().T.shape[0]).start()
    args = model.W.get_value().T.argsort()
    # pdb.set_trace()
    w_t = model.W.get_value().T
    words = []
    for i in range(model.W.get_value().T.shape[0]):
        # bar.update(i)
        W_values = []
        for n in range(-1, -11, -1):
            if dic[str(args[i][n])] not in words:
                words.append(dic[str(args[i][n])])
            # W_values.append('%s - %.2f' % (dic[str(args[i][n])], w_t[i][args[i][n]]))
            W_values.append('%s' % (dic[str(args[i][n])]))
        out.write(','.join(W_values) + '\n')
    out.close()
    print model_dirs['STEP1_analysis'] + str(len(words))

def test_phase1_sda(params, model_dirs):
    model = cPickle.load(open(model_dirs['STEP1']))
    dic = json.load(codecs.open('dataset/dataset/chi2-result-unified.rdic', 'r', 'utf-8'))
    threshold = 0.2
    max_len = 10

    out = codecs.open(model_dirs['STEP1_analysis'], 'w', 'shift-jis')
    bar = ProgressBar(maxval=model.params[0].get_value().T.shape[0]).start()
    args1 = model.params[0].get_value().T.argsort()
    w_t1 = model.params[0].get_value().T

    for i in range(model.params[0].get_value().T.shape[0]):
        bar.update(i)
        W_values = []
        for n in range(-1, -11, -1):
            W_values.append('%s - %.2f' % (dic[str(args1[i][n])], w_t1[i][args1[i][n]]))
            # W_values.append('%s' % (dic[str(args1[i][n])]))
        out.write(','.join(W_values) + '\n')
    out.close()
    
    out2 = codecs.open(model_dirs['STEP1_analysis_2'], 'w', 'shift-jis')
    bar = ProgressBar(maxval=model.params[2].get_value().T.shape[0]).start()
    args2 = model.params[2].get_value().T.argsort()
    w_t2 = model.params[2].get_value().T

    for i in range(model.params[2].get_value().T.shape[0]):
        bar.update(i)
        W_values = []
        for n in range(-1, -4, -1):
            index = args2[i][n]
            for m in range(-1, -5, -1):
                W_values.append('%s - %.2f' % (dic[str(args1[index][m])], w_t1[index][args1[index][m]]))
            # W_values.append('%s' % (dic[str(args1[i][n])]))
            W_values.append('')
        out2.write(','.join(W_values) + '\n')
    out2.close()


def main(params):
    
    default_model_dir_prefix = 'experiment/Model/'
    # default_model_dir_suffix = '/h%d_lr%s_b%s_c%s.%s' % (params['n_hidden'], str(params['learning_rate']), str(params['reg_weight']), str(params['corruption_level']), params['model'])
    # default_model_dir_suffix_csv = '/h%d_lr%s_b%s_c%s_%s.csv' % (params['n_hidden'], str(params['learning_rate']), str(params['reg_weight']), str(params['corruption_level']), params['model'])
    default_model_dir_suffix = '/h%d_lr%s_s%s_b%s_%s.%s' % (params['n_hidden'], str(params['learning_rate']), str(params['reg_weight']), str(params['batch_size']), params['dataset'], params['model'])
    # default_model_dir_suffix_csv = '/h%d_lr%s_s%s_b%s_%s.csv' % (params['n_hidden'], str(params['learning_rate']), str(params['reg_weight']), str(params['batch_size']), params['model'])
    default_model_dir_suffix_csv = '/h%d_lr%s_s%s_b%s_%s_%s.csv' % (params['n_hidden'], str(params['learning_rate']), str(params['reg_weight']), str(params['batch_size']), params['dataset'], params['model'])
    default_model_dir_suffix_csv_2 = '/h%d_lr%s_s%s_b%s_%s_2.csv' % (params['n_hidden'], str(params['learning_rate']), str(params['reg_weight']), str(params['batch_size']), params['model'])
    model_dirs = {
        'STEP1' : default_model_dir_prefix + 'STEP1' + default_model_dir_suffix,
        'STEP1_analysis' : default_model_dir_prefix + 'STEP1_analysis' + default_model_dir_suffix_csv,
        'STEP1_analysis_2' : default_model_dir_prefix + 'STEP1_analysis' + default_model_dir_suffix_csv_2,
        'STEP2' : default_model_dir_prefix + 'STEP2' + default_model_dir_suffix,
        'STEP3' : default_model_dir_prefix + 'STEP3' + default_model_dir_suffix
    }
    if params['model'] == 'sda':
        test_phase1_sda(params, model_dirs)
    else:
        test_phase1(params, model_dirs)

if __name__ == '__main__':

    if len(sys.argv) <= 1:
        params = {}
        n_hiddens = [100, 1000, 2000]
        learning_rates = [0.05]
        reg_weights = [0., 0.1, 0.01, 0.02, 0.05, 0.001, 0.0001]
        batch_sizes = [10, 20, 50, 100]
        models = ['sae', 'rbm']
        datasets = ['article', 'sentence']
        for n_hidden in n_hiddens:
            for learning_rate in learning_rates:
                for reg_weight in reg_weights:
                    for batch_size in batch_sizes:
                        for model in models:
                            for dataset in datasets:
                                params['n_hidden'] = n_hidden
                                params['learning_rate'] = learning_rate
                                params['reg_weight'] = reg_weight
                                # params['corruption_level'] = corruption_level
                                params['batch_size'] = batch_size
                                params['model'] = model
                                params['dataset'] = dataset
                                try:
                                    main(params)
                                except IOError:
                                    pass
    else:
        params = {
            'n_hidden' : int(sys.argv[1]),
            'learning_rate' : float(sys.argv[2]),
            'reg_weight' : float(sys.argv[3]),
            'batch_size' : int(sys.argv[4]),
            'model' : sys.argv[5]
        }
        main(params)
    

