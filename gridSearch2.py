# coding: utf-8

import cPickle, json, pdb, pickle, theano, sys, numpy, datetime
import copy
import os.path
import theano.tensor as T
from dataset.Nikkei import Nikkei
from experiment.CompressSparseVector.SparseAutoencoder import SparseAutoencoder, train_sae
from experiment.CompressSparseVector.RBM import RBM, train_rbm
# from experiment.PredictPrices.SdA_theano import SdA, train_SdA, pretrain_SdA, finetune_SdA
from experiment.PredictPrices.SdA_RNN import SdA, train_SdA, pretrain_SdA, finetune_SdA
from nikkei225 import getNikkei225
from experiment.PredictPrices.DBN_regression import DBN, train_DBN, pretrain_DBN, finetune_DBN
# theano.config.floatX = 'float64'

default_model_dir = '/home/fujikawa/StockPredict/src/deeplearning/experiment/Model'

params = {
    # 'dataset_type' : 'chi2_selected',
    'dataset_type' : 'all',
    'STEP1' : {
        'beta' : 0,
        'model' : 'rbm',
        'n_hidden' : 10000,
        'learning_rate' : 0.05
    },
    'STEP2' : {
        # 'experiment_type' : 'baseline'
        'experiment_type' : 'proposed'
    },
    'STEP3' : {
        'brandcode' : '0101'
    },
    'STEP4' : {
        'model' : 'dbn_r',
        'corruption_levels' : [.3, .3, .4],
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
params['STEP3']['brandcode'] = ['7203', '6758', '6502', '7201', '6501', '6702', '6753', '9984', '8058', '8031']
# params['STEP3']['brandcode'] = getNikkei225()
# params['STEP3']['brandcode'].reverse()
# params['STEP3']['brandcode'] = ['0101']
# params['STEP3']['brandcode'] = ['4324', '9984', '9433', '9613', '9983']
params['STEP4']['hidden_layers_sizes'] = [
    # [params['STEP1']['n_hidden'] / 2],
    [params['STEP1']['n_hidden'], params['STEP1']['n_hidden'] / 2]
    # [params['STEP1']['n_hidden'] , params['STEP1']['n_hidden'], params['STEP1']['n_hidden'] / 2]
]
params['STEP4']['pretrain'] = {
    'batch_size' : [50],
    'learning_rate' : [0.05],
    'epochs' : [100]
}
params['STEP4']['finetune'] = {
    'batch_size' : [50],
    'learning_rate' : [0.0001],
    'epochs' : [100]
}

model_dirs = {}

def reload_model_dirs(brandcode):
    model_dirs = {
        'STEP1' : '%s/%s/h%d_lr%.2f_b%.2f.%s.params' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP2' : '%s/%s/h%d_lr%.2f_b%.2f.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP3' : '%s/%s/%sh%d_lr%.2f_b%.2f.%s' % (default_model_dir, 'STEP3', brandcode, params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP4' : '%s/%s/%s' % (default_model_dir, 'STEP4', 'sda.pkl'),
        'STEP4_logs' : '%s/%s/h%d_lr%.2f_b%.2f.%s.%s.csv' % (default_model_dir, 'STEP4_logs', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model'], params['STEP4']['model'])
    }
    if params['STEP2']['experiment_type'] == 'baseline':
        model_dirs['STEP2'] += '.baseline'
        model_dirs['STEP3'] += '.baseline'
        model_dirs['STEP4_logs'] += '.baseline.csv'
    if params['dataset_type'] == 'chi2_selected':
        model_dirs['STEP4_logs'] = default_model_dir + '/STEP4_logs/baseline_chi2_selected'
    return model_dirs

def reguralize_data(dataset):
    for datatype in ['train', 'valid', 'test']:
        # dataset.phase2[datatype]['x'] = ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0)) ** 2) / ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0)) ** 2).max(axis=0)
        # dataset.phase2[datatype]['x'] = ((dataset.phase2[datatype]['x'] / dataset.phase2[datatype]['x'].max(axis=0)) ** 2)
        dataset.phase2[datatype]['x'] = ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0)) ** 2) / ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0) + 0.001) ** 2).max(axis=0)
        dataset.phase2[datatype]['y'] /= dataset.phase2[datatype]['y'].max()
        if theano.config.floatX == 'float32':
            print 'cast to 32bit matrix'
            dataset.phase2[datatype]['x'] = dataset.phase2[datatype]['x'].astype(numpy.float32)
            dataset.phase2[datatype]['y'] = dataset.phase2[datatype]['y'].astype(numpy.float32)

def get_model_params(model):
    params = []
    for param in model.params:
        params.append(param.get_value())
    return params

def set_model_params(model, params):
    for i, param in enumerate(model.params):
        model.params[i].set_value(params[i])
    return model

if __name__ == '__main__':
    ###   銘柄数種について実験
    
    all_size = len(params['STEP3']['brandcode']) * len(params['STEP4']['hidden_layers_sizes']) * len(params['STEP4']['pretrain']['batch_size']) * len(params['STEP4']['pretrain']['learning_rate']) * len(params['STEP4']['pretrain']['epochs']) * len(params['STEP4']['finetune']['batch_size']) * len(params['STEP4']['finetune']['learning_rate']) * len(params['STEP4']['finetune']['epochs'])
    i = 0
    for brandcode in params['STEP3']['brandcode']:
        model_dirs = reload_model_dirs(brandcode)
        dataset = None
        if params['dataset_type'] == 'all':
            dataset = ""
            dataset = cPickle.load(open(model_dirs['STEP2']))
            dataset.unify_stockprices(dataset=dataset.unified, brandcode=brandcode, dataset_type=params['dataset_type'])
            reguralize_data(dataset)
        else:
            dataset = Nikkei(dataset_type=params['dataset_type'], brandcode=brandcode)
            dataset.unify_stockprices(dataset=dataset.raw_data[brandcode], brandcode=brandcode, dataset_type=params['dataset_type'])
            # dataset.unify_stockprices(dataset.raw_data[brandcode])
        for hidden_layers_sizes in params['STEP4']['hidden_layers_sizes']:
            for batch_size_pretrain in params['STEP4']['pretrain']['batch_size']:
                for learning_rate_pretrain in params['STEP4']['pretrain']['learning_rate']:
                    for epochs_pretrain in params['STEP4']['pretrain']['epochs']:
                        # dataset.phase2_input_size = 5000
                        pretrain_model = ""
                        pretrain_model = pretrain_SdA(dataset=dataset, 
                            hidden_layers_sizes=hidden_layers_sizes,
                            pretrain_lr=learning_rate_pretrain,
                            pretrain_batch_size=batch_size_pretrain,
                            pretrain_epochs=epochs_pretrain
                        )
                        pretrain_params = get_model_params(pretrain_model)

                        for batch_size_finetune in params['STEP4']['finetune']['batch_size']:
                            for learning_rate_finetune in params['STEP4']['finetune']['learning_rate']:
                                for epochs_finetune in params['STEP4']['finetune']['epochs']:
                                    
                                    # pdb.set_trace()
                                    # print dataset.phase2_input_size
                                    # pretrain_model_copy = copy.copy(pretrain_model)
                                    set_model_params(pretrain_model, pretrain_params)
                                    finetune_model = ""
                                    finetune_model, best_validation_loss, test_score = finetune_SdA(dataset=dataset, 
                                        sda = pretrain_model,
                                        finetune_lr=learning_rate_finetune,
                                        finetune_batch_size=batch_size_finetune,
                                        finetune_epochs=epochs_finetune
                                    )

                                    i += 1
                                    print '%d / %d is done...' % (i , all_size)
                                    out = open(model_dirs['STEP4_logs'], 'a')
                                    out.write('%f,%f,%s,%s,%d,%f,%d,%d,%f,%d,%s\n' % (best_validation_loss, test_score, brandcode, str(hidden_layers_sizes).replace(',', ' '), batch_size_pretrain, learning_rate_pretrain, epochs_pretrain, batch_size_finetune, learning_rate_finetune, epochs_finetune, str(datetime.datetime.now())))
                                    out.close()



