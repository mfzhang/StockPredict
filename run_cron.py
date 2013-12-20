# coding: utf-8

import cPickle, json, pdb, pickle, theano, sys
import os.path
import theano.tensor as T
from dataset.Nikkei import Nikkei
from experiment.CompressSparseVector.SparseAutoencoder import SparseAutoencoder, train_sae
from experiment.CompressSparseVector.RBM import RBM, train_rbm
from experiment.PredictPrices.SdA_theano import SdA, train_SdA



default_model_dir = '/home/fujikawa/StockPredict/src/deeplearning/experiment/Model'
upload_log_dir = default_model_dir + '/STEP1/upload_log.json'

params = {
    'dataset_type' : 'all',
    'STEP1' : {
        'beta' : 1.,
        'model' : 'sda',
        'n_hidden' : 1000,
        'learning_rate' : 0.05
    },
    'STEP3' : {
        'brandcode' : '0101'
    },
    'STEP4' : {
        'corruption_levels' : [.1, .2, .3],
        'hidden_layers_sizes' : [1000, 500, 500],
        'pretrain' : {
            'batch_size' : 20,
            'learning_rate' : 0.05,
            'epochs' : 50
        },
        'finetune' : {
            'batch_size' : 20,
            'learning_rate' : 0.1,
            'epochs' : 50
        }
    }
}

model_dirs = {
    'STEP1' : '%s/%s/h%d_lr%f_b%f.%s.params' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
    'STEP2' : '%s/%s/h%d_lr%f_b%f.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
    'STEP3' : '%s/%s/%sh%d_lr%f_b%f.%s' % (default_model_dir, 'STEP3', params['STEP3']['brandcode'], params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
    'STEP4' : '%s/%s/%s' % (default_model_dir, 'STEP4', 'sda.pkl')
}


def load_model(input=None, params_dir=None):
    params = cPickle.load(open(params_dir))
    model = SparseAutoencoder(input=input, params=params)
    return model

def reload_model_dirs(params):
    model_dirs = {
        'STEP1' : '%s/%s/h%d_lr%f_b%f.%s.params' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP2' : '%s/%s/h%d_lr%f_b%f.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP3' : '%s/%s/%sh%d_lr%f_b%f.%s' % (default_model_dir, 'STEP3', params['STEP3']['brandcode'], params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP4' : '%s/%s/%s' % (default_model_dir, 'STEP4', 'sda.pkl')
    }
    return model_dirs


##################################################################################
###  STEP 2: 前のステップで訓練された Sparse Auto-encoderを用いた複数記事の圧縮表現の獲得  ###
##################################################################################

def unify_kijis(dataset):
    print 'STEP 2 start...'
    if dataset == None:
        print 'dataset load...'
        dataset = Nikkei(dataset_type=params['dataset_type'], brandcode=params['STEP3']['brandcode'])
    # model = load_model(model_dirs['STEP1'])
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    ###########################################################
    model = load_model(input=x, params_dir=model_dirs['STEP1'])
    # model = cPickle.load(open(model_dirs['STEP1']))
    ###########################################################
    dataset.unify_kijis(model)
    out = open(model_dirs['STEP2'], 'w')
    out.write(cPickle.dumps(dataset))
    return dataset

######################################################
#####          PHASE2: 各銘柄の株価の予測            #####
######################################################
###  STEP 3: 指定された銘柄の株価と記事データを組み合わせる  ###
######################################################

def unify_stockprices(dataset):
    print 'STEP 3 start...'
    if dataset == None:
        print 'dataset load...'
        dataset = cPickle.load(open(model_dirs['STEP2']))

    dataset.unify_stockprices(dataset=dataset.unified, brandcode=params['STEP3']['brandcode'], dataset_type=params['dataset_type'])
    out = open(model_dirs['STEP3'], 'w')
    out.write(cPickle.dumps(dataset))
    return dataset

if __name__ == '__main__':
	upload_log = {}
	if os.path.exists(upload_log_dir)
		upload_log = json.load(open(upload_log_dir))
	params['STEP1']['beta'] = beta
    params['STEP1']['n_hidden'] = n_hidden
    params['STEP1']['learning_rate'] = learning_rate
    model_dirs = reload_model_dirs(params)
    dataset=None
    unify_kijis(dataset)
    unify_stockprices(dataset)
    predict(dataset)




