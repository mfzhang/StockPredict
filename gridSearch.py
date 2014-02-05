# coding: utf-8

import cPickle, json, pdb, pickle, theano, sys, datetime
import copy
import numpy as np
import os.path
import theano.tensor as T
from dataset.Nikkei import Nikkei
from experiment.CompressSparseVector.SparseAutoencoder import SparseAutoencoder, train_sae
from experiment.CompressSparseVector.RBM import RBM, train_rbm
# from experiment.PredictPrices.SdA_theano import SdA, train_SdA, pretrain_SdA, finetune_SdA
from experiment.PredictPrices import SdA
from experiment.PredictPrices import SdA_RNN
from experiment.PredictPrices import DBN
from run import *


from nikkei225 import getNikkei225

# theano.config.floatX = 'float64'
# from run import *

default_model_dir = '/home/fujikawa/StockPredict/src/deeplearning/experiment/Model'

params = {
    # 'experiment_type' : 'chi2_selected',
    'dataset_type' : 'article',
    'label_type' : 1,
    'experiment_type' : 'average',
    'activation_function' : 'ReLU',
    # 'experiment_type' : 'max',
    'STEP1' : {
        'reg_weight' : 0.0,
        'model' : 'rbm',
        'n_hidden' : 1000,
        'learning_rate' : 0.05,
        'batch_size' : 20
    },
    'STEP3' : {
        'brandcode' : '0101'
    },
    'STEP4' : {
        'dropout' : False,
        'recurrent' : False,
        'model' : 'sda_4_dropout_recurrent',
        'hidden_recurrent' : 500,
        'corruption_levels' : [.3, .3, .4],
        'k' : 1,
        'hidden_layers_sizes' : [1000, 500, 500],
        'pretrain' : {
            'batch_size' : 20,
            'learning_rate' : 1e-2,
            'epochs' : 100
        },
        'finetune' : {
            'batch_size' : 20,
            'learning_rate' : 1e-2,
            'epochs' : 100
        }
    }
}

###   grid searchのパラメータ格納
params['STEP3']['brandcode'] = ['0101', '7203', '6758', '6502', '7201', '6501', '6702', '6753', '8058', '8031', '7751']
# params['STEP3']['brandcode'] = ['0101']
# params['STEP3']['brandcode'] = getNikkei225()
ALL_brandcodes = getNikkei225()
NG_brandcodes = ['2768', '3382', '3893', '4188', '4324', '4568', '4689', '4704', '5411', '6674', '8303', '8306', '8308', '8309', '8316', '8411', '8766', '8795', '9983', '6796', '9984', '6366', '2282', '7004', '7013', '9020', '9432']
brandcodes = list(set(ALL_brandcodes) - set(NG_brandcodes))
params['STEP3']['brandcode'] = brandcodes


params['STEP4']['hidden_layers_sizes'] = [
    [500, 500]
    

    # [500],
    # [1000, 500]
    # [params['STEP1']['n_hidden'] , params['STEP1']['n_hidden'], params['STEP1']['n_hidden'] / 2]
]
params['STEP4']['pretrain'] = {
    'batch_size' : [50],
    'learning_rate' : [1e-6],
    'epochs' : [300]
}


params['STEP4']['finetune'] = {
    'batch_size' : [50, 100],
    'learning_rate' : [1e-2, 1e-3],
    'epochs' : [300]
}


if params['experiment_type'] == 'chi2_selected':
    model_dirs['STEP4_logs'] = default_model_dir + '/STEP4_logs/baseline_chi2_selected'



if __name__ == '__main__':
    ###   銘柄数種について実験

    if len(sys.argv) < 6:
        print len(sys.argv)
        print 'args invalid'
        sys.exit()
    else:
        params['STEP4']['model'] = '%s_%s_%s_%s' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        params['label_type'] = int(sys.argv[2])
        if sys.argv[3] == 'dropout':
            params['STEP4']['dropout'] = True
        if sys.argv[4] == 'recurrent':
            params['STEP4']['recurrent'] = True
        params['experiment_type'] = sys.argv[5]

    model_dirs = {
        'STEP1' : '%s/%s/h%d_lr%s_s%s_b%s_%s.%s' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['batch_size']), params['dataset_type'], params['STEP1']['model']),
        'STEP2' : '%s/%s/h%d_lr%s_s%s_b%s_%s.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['batch_size']), params['dataset_type'], params['STEP1']['model']),
        'STEP3' : '%s/%s/h%d_lr%s_s%s_b%s_%s.%s' % (default_model_dir, 'STEP3', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['batch_size']), params['dataset_type'], params['STEP1']['model']),
        'STEP3_logs' : '%s/%s/h%d_lr%s_s%s_b%s_%s_%s_%s_%s_%s.csv' % (default_model_dir, 'STEP3_logs', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['batch_size']), params['dataset_type'], params['activation_function'], params['STEP1']['model'], params['STEP4']['model'], params['experiment_type']),    
    }


    print model_dirs['STEP3_logs']
    if os.path.exists(model_dirs['STEP3_logs']):
        print 'ファイルが存在します．'
        # sys.exit()
    else:
        out = open(model_dirs['STEP3_logs'], 'w')
        out.write('best_validation_loss,test_score,brandcode,hidden_layers_sizes,batch_size_pretrain,learning_rate_pretrain,epochs_pretrain,batch_size_finetune,learning_rate_finetune,epochs_finetune,best_epoch,datetime\n')
        out.close()
    
    
    
    # model_dirs = reload_model_dirs()
    
    
    if params['STEP4']['model'].split('_')[0] == 'sda':
        model = SdA

    elif params['STEP4']['model'].split('_')[0] == 'dbn':
        model = DBN
        
    else:
        sys.exit()

    all_size = len(params['STEP3']['brandcode']) * len(params['STEP4']['hidden_layers_sizes']) * len(params['STEP4']['pretrain']['batch_size']) * len(params['STEP4']['pretrain']['learning_rate']) * len(params['STEP4']['pretrain']['epochs']) * len(params['STEP4']['finetune']['batch_size']) * len(params['STEP4']['finetune']['learning_rate']) * len(params['STEP4']['finetune']['epochs'])
    i = 0
   
    label_type = params['label_type']
    if params['experiment_type'] == 'baseline':
        print 'start to load baseline dataset...'
        dataset = cPickle.load(open(default_model_dir + '/STEP2/baseline_original'))
        print 'start to unify stockprice...'
        dataset.unify_stockprices(dataset=dataset.baseline, brandcodes=params['STEP3']['brandcode'], dataset_type=params['experiment_type'], label_type=label_type, y_type=get_y_type(label_type), y_force_list=params['STEP4']['recurrent'])
    else:
        print 'start to load proposed dataset...'
        dataset = cPickle.load(open(model_dirs['STEP2']))
        if params['experiment_type'] == 'average':
            print 'start to unify stockprice (average pooling)...'
            usedata = dataset.unified_mean
        else:
            print 'start to unify stockprice (max pooling)...'
            usedata = dataset.unified_max
        dataset.unify_stockprices(dataset=usedata, brandcodes=params['STEP3']['brandcode'], dataset_type=params['experiment_type'], label_type=label_type, y_type=get_y_type(label_type), y_force_list=params['STEP4']['recurrent'])

    if params['experiment_type'] != 'baseline':
        reguralize_data(dataset, params['STEP3']['brandcode'])
    optimizeGPU(dataset, params['STEP3']['brandcode'])
    # if params['experiment_type'] == 'chi2_selected':
    #     dataset = Nikkei(dataset_type=params['experiment_type'], brandcode=brandcode)
    #     dataset.unify_stockprices(dataset=dataset.raw_data[brandcode], brandcode=brandcode, dataset_type=params['experiment_type'])
    # else:
    #     dataset = ""
    #     dataset = cPickle.load(open(model_dirs['STEP2']))
    #     dataset.unify_stockprices(dataset=dataset.unified, brandcodes=params['STEP3']['brandcode'], dataset_type=params['experiment_type'], label_type=params['label_type'])
    #     reguralize_data(dataset, params['STEP3']['brandcode'])
    
    startnum = -1
    
    for hidden_layers_sizes in params['STEP4']['hidden_layers_sizes']:
        for batch_size_pretrain in params['STEP4']['pretrain']['batch_size']:
            for learning_rate_pretrain in params['STEP4']['pretrain']['learning_rate']:
                for epochs_pretrain in params['STEP4']['pretrain']['epochs']:
                    change_brand(dataset, '0101')
                    
                    # dataset.phase2_input_size = 5000
                    pretrain_model = ""
                    pretrain_params = {
                        'dataset' : dataset, 
                        'hidden_layers_sizes' : hidden_layers_sizes,
                        'pretrain_lr' : learning_rate_pretrain,
                        'pretrain_batch_size' : batch_size_pretrain,
                        'pretrain_epochs' : epochs_pretrain,
                        'corruption_levels' : params['STEP4']['corruption_levels'],
                        'k' : params['STEP4']['k'],
                        'y_type' : get_y_type(label_type),
                        'n_outs' : get_y_type(label_type) + 1,
                        'gbrbm' : params['experiment_type'] != 'baseline',
                        'recurrent' : params['STEP4']['recurrent'],
                        'dropout' : params['STEP4']['dropout'],
                        'activation_function' : params['activation_function']
                    }
                    print pretrain_params
                    pretrain_model = model.pretrain(pretrain_params)
                    pre_params = get_model_params(pretrain_model)

                    for batch_size_finetune in params['STEP4']['finetune']['batch_size']:
                        for learning_rate_finetune in params['STEP4']['finetune']['learning_rate']:
                            for epochs_finetune in params['STEP4']['finetune']['epochs']:
                                finetune_params = {
                                    'dataset' : dataset,
                                    'model' : pretrain_model,
                                    'finetune_lr' : learning_rate_finetune,
                                    'finetune_batch_size' : batch_size_finetune,
                                    'finetune_epochs' : epochs_finetune,
                                    'y_type' : get_y_type(label_type)
                                }
                                print finetune_params
                                
                                for brandcode in params['STEP3']['brandcode']:
                                    ##  y に 各銘柄の正解データを格納
                                    print brandcode
                                    if i > startnum:
                                        change_brand(dataset, brandcode)
                                        set_model_params(pretrain_model, pre_params)
                                        finetune_model = ""
                                        
                                        finetune_model, best_validation_loss, test_score, best_epoch = model.finetune(finetune_params)
                                        
                                        
                                        print '%d / %d is done...' % (i , all_size)
                                        out = open(model_dirs['STEP3_logs'], 'a')
                                        out.write('%f,%f,%s,%s,%d,%f,%d,%d,%f,%d,%d,%s\n' % (best_validation_loss, test_score, brandcode, str(hidden_layers_sizes).replace(',', ' '), batch_size_pretrain, learning_rate_pretrain, epochs_pretrain, batch_size_finetune, learning_rate_finetune, epochs_finetune, best_epoch, str(datetime.datetime.now())))
                                        out.close()
                                    i += 1



