# coding: utf-8

import cPickle, json, pdb, pickle, theano, sys, numpy
import os.path
import theano.tensor as T
from dataset.Nikkei import Nikkei
from yoshihara.PredictPrices.stackRnnRbm import stackRnnRbm, train_rnnrbm
#import run
# theano.config.floatX = 'float64'

default_model_dir = 'yoshihara/Model'

params = {
    'dataset_type' : 'chi2_selected',
    'STEP1' : {
        'beta' : 0,
        'model' : 'rnnrbm',
        'n_hidden' : 1000,
        'learning_rate' : 0.05
    },
    'STEP3' : {
        'brandcode' : '0101'
    },
    'STEP4' : {
        'hidden_layers_sizes' : [2000,1500],
        'hidden_recurrent': 1000,
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
params['STEP3']['brandcode'] = ['0101']
params['STEP4']['hidden_layers_sizes'] = [
    [params['STEP1']['n_hidden'] * 2, params['STEP1']['n_hidden']*1.5]
]
params['STEP4']['hidden_recurrent'] = [10, 100, 500, 1000]
params['STEP4']['pretrain'] = {
    'batch_size' : [30, 50, 100],
    'learning_rate' : [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'epochs' : [50]
}
params['STEP4']['finetune'] = {
    'batch_size' : [50, 100],
    'learning_rate' : [0.1, 1.0, 2.0],
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
        'STEP3' : '%s/%s/%sh%d_%s' % (default_model_dir, 'STEP3', brandcode, params['STEP1']['n_hidden'], params['STEP1']['model']),
        'STEP4' : '%s/%s/%s' % (default_model_dir, 'STEP4', 'rnnrbm.pkl'),
        'STEP4_logs' : '%s/%s/%sh%d_%s.log' % (default_model_dir, 'STEP4_logs', brandcode, params['STEP1']['n_hidden'], params['STEP1']['model'])
    }
    return model_dirs

if __name__ == '__main__':
    ###   銘柄数種について実験
    all_size = len(params['STEP3']['brandcode']) * len(params['STEP4']['hidden_layers_sizes']) * len(params['STEP4']['pretrain']['batch_size']) * len(params['STEP4']['pretrain']['learning_rate']) * len(params['STEP4']['pretrain']['epochs']) * len(params['STEP4']['finetune']['batch_size']) * len(params['STEP4']['finetune']['learning_rate']) * len(params['STEP4']['finetune']['epochs'])
    i = 0
    for brandcode in params['STEP3']['brandcode']:
        model_dirs = reload_model_dirs(brandcode)
        dataset = Nikkei(dataset_type=params['dataset_type'], brandcode=brandcode)
        dataset.unify_stockprices(dataset.raw_data[brandcode])
        for hidden_layers_sizes in params['STEP4']['hidden_layers_sizes']:
            for hidden_recurrent in params['STEP4']['hidden_recurrent']:
                for batch_size_pretrain in params['STEP4']['pretrain']['batch_size']:
                    for learning_rate_pretrain in params['STEP4']['pretrain']['learning_rate']:
                        for epochs_pretrain in params['STEP4']['pretrain']['epochs']:
                            for batch_size_finetune in params['STEP4']['finetune']['batch_size']:
                                for learning_rate_finetune in params['STEP4']['finetune']['learning_rate']:
                                    for epochs_finetune in params['STEP4']['finetune']['epochs']:
                                        result = train_rnnrbm(dataset=dataset, 
                                            hidden_layers_sizes=hidden_layers_sizes,
                                            hidden_recurrent=hidden_recurrent,
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
                                        out.write('%f,%s,%s,%s,%d,%f,%d,%d,%f,%d\n' % (result, brandcode, str(hidden_layers_sizes), hidden_recurrent, batch_size_pretrain, learning_rate_pretrain, epochs_pretrain, batch_size_finetune, learning_rate_finetune, epochs_finetune))
                                        out.close()



