# coding: utf-8

import cPickle, json, pdb, pickle, theano, sys, numpy, datetime
import copy
import os.path
import theano.tensor as T
from dataset.Nikkei import Nikkei
from experiment.CompressSparseVector.SparseAutoencoder import SparseAutoencoder, train_sae
from experiment.CompressSparseVector.RBM import RBM, train_rbm
# from experiment.PredictPrices.SdA_theano import SdA, train_SdA, pretrain_SdA, finetune_SdA
from experiment.PredictPrices import SdA_regression
from experiment.PredictPrices import SdA_RNN
from experiment.PredictPrices import DBN_regression



from nikkei225 import getNikkei225

# theano.config.floatX = 'float64'
# from run import *

default_model_dir = '/home/fujikawa/StockPredict/src/deeplearning/experiment/Model'

params = {
    # 'experiment_type' : 'chi2_selected',
    'label_type' : 1,
    'experiment_type' : 'all',
    'STEP1' : {
        'beta' : 0.001,
        'model' : 'rbm',
        'n_hidden' : 2500,
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
params['STEP3']['brandcode'] = ['0101', '7203', '6758', '6502', '7201', '6501', '6702', '6753', '8058', '8031', '7751']
# params['STEP3']['brandcode'] = getNikkei225()
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

model_dirs = {
    'STEP1' : '%s/%s/h%d_lr%s_b%s.%s.params' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['beta']), params['STEP1']['model']),
    'STEP2' : '%s/%s/h%d_lr%s_b%s.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['beta']), params['STEP1']['model']),
    'STEP3' : '%s/%s/h%d_lr%s_b%s.%s' % (default_model_dir, 'STEP3', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['beta']), params['STEP1']['model']),
    'STEP3_logs' : '%s/%s/h%d_lr%s_b%s_%s_%s_%s.csv' % (default_model_dir, 'STEP3_logs', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['beta']), params['STEP1']['model'], params['STEP4']['model'], params['experiment_type'])
}
if params['experiment_type'] == 'chi2_selected':
    model_dirs['STEP4_logs'] = default_model_dir + '/STEP4_logs/baseline_chi2_selected'



# def reload_model_dirs():
#     model_dirs = {
#         'STEP1' : '%s/%s/h%d_lr%s_b%s.%s.params' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['beta']), params['STEP1']['model']),
#         'STEP2' : '%s/%s/h%d_lr%s_b%s.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['beta']), params['STEP1']['model']),
#         'STEP3' : '%s/%s/h%d_lr%s_b%s.%s' % (default_model_dir, 'STEP3', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['beta']), params['STEP1']['model']),
#         'STEP3_logs' : '%s/%s/h%d_lr%s_b%s_%s_%s_%s.csv' % (default_model_dir, 'STEP3_logs', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['beta']), params['STEP1']['model'], params['STEP4']['model'], params['experiment_type'])
#     }
#     if params['experiment_type'] == 'chi2_selected':
#         model_dirs['STEP4_logs'] = default_model_dir + '/STEP4_logs/baseline_chi2_selected'
#     return model_dirs

def reguralize_data(dataset, brandcodes):
    for datatype in ['train', 'valid', 'test']:
        # pdb.set_trace()
        # dataset.phase2[datatype]['x'] = ((dataset.phase2[datatype]['x'] / dataset.phase2[datatype]['x'].max(axis=0)) ** 2)
        dataset.phase2[datatype]['x'] = ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0)) ** 2) / ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0) + 0.001) ** 2).max(axis=0)
        for brandcode in brandcodes:
            dataset.phase2[datatype][brandcode] /= dataset.phase2[datatype][brandcode].max()
        if theano.config.floatX == 'float32':
            print 'cast to 32bit matrix'
            dataset.phase2[datatype]['x'] = dataset.phase2[datatype]['x'].astype(numpy.float32)
            for brandcode in brandcodes:
                dataset.phase2[datatype][brandcode] = dataset.phase2[datatype][brandcode].astype(numpy.float32)

def change_brand(dataset, brandcode):
    for datatype in ['train', 'valid', 'test']:
        dataset.phase2[datatype]['y'] = dataset.phase2[datatype][brandcode]

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
    
    # model_dirs = reload_model_dirs()
    dataset = None
    if params['experiment_type'] == 'chi2_selected':
        dataset = Nikkei(dataset_type=params['experiment_type'], brandcode=brandcode)
        dataset.unify_stockprices(dataset=dataset.raw_data[brandcode], brandcode=brandcode, dataset_type=params['experiment_type'])
    else:
        dataset = ""
        dataset = cPickle.load(open(model_dirs['STEP2']))
        dataset.unify_stockprices(dataset=dataset.unified, brandcodes=params['STEP3']['brandcode'], dataset_type=params['experiment_type'], label_type=params['label_type'])
        reguralize_data(dataset, params['STEP3']['brandcode'])
    if params['STEP4']['model'] == 'sda_r':
        model = SdA_regression
    elif params['STEP4']['model'] == 'dbn_r':
        model = DBN_regression
    for hidden_layers_sizes in params['STEP4']['hidden_layers_sizes']:
        for batch_size_pretrain in params['STEP4']['pretrain']['batch_size']:
            for learning_rate_pretrain in params['STEP4']['pretrain']['learning_rate']:
                for epochs_pretrain in params['STEP4']['pretrain']['epochs']:
                    change_brand(dataset, '0101')
                    
                    # dataset.phase2_input_size = 5000
                    pretrain_model = ""
                    pretrain_model = model.pretrain(
                        dataset=dataset, 
                        hidden_layers_sizes=hidden_layers_sizes,
                        pretrain_lr=learning_rate_pretrain,
                        pretrain_batch_size=batch_size_pretrain,
                        pretrain_epochs=epochs_pretrain
                    )
                    pretrain_params = get_model_params(pretrain_model)

                    for batch_size_finetune in params['STEP4']['finetune']['batch_size']:
                        for learning_rate_finetune in params['STEP4']['finetune']['learning_rate']:
                            for epochs_finetune in params['STEP4']['finetune']['epochs']:
                                for brandcode in params['STEP3']['brandcode']:
                                    ##  y に 各銘柄の正解データを格納
                                    change_brand(dataset, brandcode)
                                    set_model_params(pretrain_model, pretrain_params)
                                    finetune_model = ""
                                    finetune_model, best_validation_loss, test_score, best_epoch = model.finetune(
                                        dataset=dataset, 
                                        model = pretrain_model,
                                        finetune_lr=learning_rate_finetune,
                                        finetune_batch_size=batch_size_finetune,
                                        finetune_epochs=epochs_finetune
                                    )

                                    i += 1
                                    print '%d / %d is done...' % (i , all_size)
                                    out = open(model_dirs['STEP3_logs'], 'a')
                                    out.write('%f,%f,%s,%s,%d,%f,%d,%d,%f,%d,%s\n' % (best_validation_loss, test_score, brandcode, str(hidden_layers_sizes).replace(',', ' '), batch_size_pretrain, learning_rate_pretrain, epochs_pretrain, batch_size_finetune, learning_rate_finetune, epochs_finetune, str(datetime.datetime.now())))
                                    out.close()



