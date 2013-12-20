# coding: utf-8

import cPickle, json, pdb, pickle, theano, sys, numpy
import os.path
import theano.tensor as T
from dataset.Nikkei import Nikkei
from experiment.CompressSparseVector.SparseAutoencoder import SparseAutoencoder, train_sae
from experiment.CompressSparseVector.RBM import RBM, train_rbm
from experiment.PredictPrices.SdA_theano import SdA, train_SdA
import run
# theano.config.floatX = 'float64'

default_model_dir = '/home/fujikawa/StockPredict/src/deeplearning/experiment/Model'

params = {
    'dataset_type' : 'all',
    'STEP1' : {
        'beta' : 3,
        'model' : 'sae',
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
            'learning_rate' : 1e-2,
            'epochs' : 50
        },
        'finetune' : {
            'batch_size' : 20,
            'learning_rate' : 1e-2,
            'epochs' : 50
        }
    }
}

###   grid searchのパラメータ格納
params['STEP3']['brandcode'] = ['0101', '7203', '4324', '9984', '9433', ]
params['STEP4']['hidden_layers_sizes'] = [
    [params['STEP1']['n_hidden'], params['STEP1']['n_hidden'] / 2, params['STEP1']['n_hidden'] / 2],
    [params['STEP1']['n_hidden'] * 1.5, params['STEP1']['n_hidden'], params['STEP1']['n_hidden']]
]
params['STEP4']['pretrain'] = {
    'batch_size' : [50, 100],
    'learning_rate' : [0.05, 0.1],
    'epochs' : [50]
}
params['STEP4']['finetune'] = {
    'batch_size' : [50, 100],
    'learning_rate' : [0.05, 0.1],
    'epochs' : [50]
}

model_dirs = {}
# model_dirs = {
#     'STEP1' : '%s/%s/h%d_lr%f.%s.params' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['model']),
#     'STEP2' : '%s/%s/h%d_lr%f.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['model']),
#     'STEP3' : '%s/%s/%sh%d_lr%f.%s' % (default_model_dir, 'STEP3', params['STEP3']['brandcode'], params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['model']),
#     'STEP4' : '%s/%s/%s' % (default_model_dir, 'STEP4', 'sda.pkl'),
#     'STEP4_logs' : '%s/%s/%sh%d_lr%f.%s.log' % (default_model_dir, 'STEP4_logs', params['STEP3']['brandcode'], params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['model'])
# }

def reload_model_dirs(brandcode):
    model_dirs = {
        'STEP1' : '%s/%s/h%d_lr%f_b%f.%s.params' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP2' : '%s/%s/h%d_lr%f_b%f.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP3' : '%s/%s/%sh%d_lr%f_b%f.%s' % (default_model_dir, 'STEP3', brandcode, params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP4' : '%s/%s/%s' % (default_model_dir, 'STEP4', 'sda.pkl'),
        'STEP4_logs' : '%s/%s/%sh%d_lr%f_b%f.%s.log' % (default_model_dir, 'STEP4_logs', brandcode, params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model'])
    }
    return model_dirs

def reguralize_data(dataset):
    for datatype in ['train', 'valid', 'test']:
        dataset.phase2[datatype]['x'] = ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0)) ** 2) / ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0)) ** 2).max(axis=0)
        if theano.config.floatX == 'float32':
            print 'cast to 32bit matrix'
            dataset.phase2[datatype]['x'] = dataset.phase2[datatype]['x'].astype(numpy.float32)
            dataset.phase2[datatype]['y'] = dataset.phase2[datatype]['y'].astype(numpy.float32)

if __name__ == '__main__':
    ###   銘柄数種について実験
    
    all_size = len(params['STEP3']['brandcode']) * len(params['STEP4']['hidden_layers_sizes']) * len(params['STEP4']['pretrain']['batch_size']) * len(params['STEP4']['pretrain']['learning_rate']) * len(params['STEP4']['pretrain']['epochs']) * len(params['STEP4']['finetune']['batch_size']) * len(params['STEP4']['finetune']['learning_rate']) * len(params['STEP4']['finetune']['epochs'])
    i = 0
    for brandcode in params['STEP3']['brandcode']:
        model_dirs = reload_model_dirs(brandcode)
        dataset = cPickle.load(open(model_dirs['STEP3']))
        reguralize_data(dataset)
        for hidden_layers_sizes in params['STEP4']['hidden_layers_sizes']:
            for batch_size_pretrain in params['STEP4']['pretrain']['batch_size']:
                for learning_rate_pretrain in params['STEP4']['pretrain']['learning_rate']:
                    for epochs_pretrain in params['STEP4']['pretrain']['epochs']:
                        for batch_size_finetune in params['STEP4']['finetune']['batch_size']:
                            for learning_rate_finetune in params['STEP4']['finetune']['learning_rate']:
                                for epochs_finetune in params['STEP4']['finetune']['epochs']:
                                    result = train_SdA(dataset=dataset, 
                                        hidden_layers_sizes=hidden_layers_sizes,
                                        corruption_levels=params['STEP4']['corruption_levels'],
                                        pretrain_lr=learning_rate_pretrain,
                                        pretrain_batch_size=batch_size_pretrain,
                                        pretrain_epochs=epochs_pretrain,
                                        finetune_lr=learning_rate_finetune,
                                        finetune_batch_size=batch_size_finetune,
                                        finetune_epochs=epochs_finetune
                                    )
                                    i += 1
                                    print '%d / %d is done...' % (i , all_size)
                                    out = open(model_dirs['STEP4_logs'], 'a')
                                    out.write('%f,%s,%s,%d,%f,%d,%d,%f,%d\n' % (result, brandcode, str(hidden_layers_sizes), batch_size_pretrain, learning_rate_pretrain, epochs_pretrain, batch_size_finetune, learning_rate_finetune, epochs_finetune))
                                    out.close()



