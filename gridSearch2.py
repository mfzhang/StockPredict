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

from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV

from nikkei225 import getNikkei225

# theano.config.floatX = 'float64'
# from run import *

default_model_dir = '/home/fujikawa/StockPredict/src/deeplearning/experiment/Model'

params = {
    # 'experiment_type' : 'chi2_selected',
    'label_type' : 1,
    'experiment_type' : 'average',
    'dataset_type' : 'article',
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
    [1000]
    

    # [500],
    # [1000, 500]
    # [params['STEP1']['n_hidden'] , params['STEP1']['n_hidden'], params['STEP1']['n_hidden'] / 2]
]
params['STEP4']['pretrain'] = {
    'batch_size' : [100, 50],
    'epochs' : [300]
}


params['STEP4']['finetune'] = {
    'batch_size' : [30],
    'learning_rate' : [1e-1, 1e-2, 5e-2, 1e-3],
    'epochs' : [100]
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
        'STEP3_logs' : '%s/%s/%s_h%d_lr%s_s%s_b%s_%s_%s_%s.csv' % (default_model_dir, 'STEP3_logs', params['STEP4']['model'], params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['batch_size']), params['dataset_type'], params['STEP1']['model'], params['experiment_type']),    
    }

    # model_dirs = {
    #     'STEP1' : '%s/%s/h%d_lr%s_s%s_b%s.%s' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['batch_size']), params['STEP1']['model']),
    #     'STEP2' : '%s/%s/h%d_lr%s_s%s_b%s.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['batch_size']), params['STEP1']['model']),
    #     'STEP3' : '%s/%s/h%d_lr%s_s%s_b%s.%s' % (default_model_dir, 'STEP3', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['batch_size']), params['STEP1']['model']),
    #     'STEP3_logs' : '%s/%s/%s_%s_idf.csv' % (default_model_dir, 'STEP3_logs', params['STEP4']['model'], params['experiment_type']),    
    # }


    print model_dirs['STEP3_logs']
    if os.path.exists(model_dirs['STEP3_logs']):
        print 'ファイルが存在します．'
        # sys.exit()
    else:
        out = open(model_dirs['STEP3_logs'], 'w')
        out.write('brandcode,train_acc,test_acc\n')
        out.close()
    
    
    
    # model_dirs = reload_model_dirs()
    
    
    if params['STEP4']['model'].split('_')[0] == 'sda':
        model = SdA
        params['STEP4']['pretrain']['learning_rate'] = [1e-5]

    elif params['STEP4']['model'].split('_')[0] == 'dbn':
        model = DBN
        params['STEP4']['pretrain']['learning_rate'] = [1e-5]
    elif params['STEP4']['model'].split('_')[0] == 'svr':
        model = SVR
    elif params['STEP4']['model'].split('_')[0] == 'svc':
        model = SVC
    else:
        sys.exit()


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
    
    def transformY(data_y):
        y = []
        for data in data_y:
            y.append(data[0])
        return np.array(y)
    train_x = dataset.phase2['train']['x']
    train_x = np.append(train_x, dataset.phase2['valid']['x'], 0)
    test_x = dataset.phase2['test']['x']
    train_x_original = train_x
    test_x_original = test_x

    startbrand = '9007'
    flag = True
    for i, brandcode in enumerate(params['STEP3']['brandcode']):
        ##  y に 各銘柄の正解データを格納
        change_brand(dataset, brandcode)
        print brandcode

        if flag:
            if get_y_type(label_type) == 0:
                ##   回帰問題の場合
                train_y = transformY(dataset.phase2['train']['y'])
                train_y = np.append(train_y, transformY(dataset.phase2['valid']['y']), 0)
                test_y = transformY(dataset.phase2['test']['y'])
            else:
                ##   分類問題の場合
                train_y = dataset.phase2['train']['y']
                train_y = np.append(train_y, dataset.phase2['valid']['y'], 0)
                test_y = dataset.phase2['test']['y']
            
            ####     各種分類アルゴリズムの詳細設定     ####   
            if model == SVR:
                tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [10**i for i in range(-4,0)], 'C': [10**i for i in range(0,4)]}]
                gscv = GridSearchCV(model(), tuned_parameters, cv=5, scoring="mean_squared_error", n_jobs=5)
                gscv.fit(train_x, train_y)
                best_model = gscv.best_estimator_
            elif model == SVC:
                tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [10**i for i in range(-4,0)], 'C': [10**i for i in range(0,4)]}]
                gscv = GridSearchCV(model(), tuned_parameters, cv=5, n_jobs=5)
                gscv.fit(train_x, train_y)
                best_model = gscv.best_estimator_
            else:
                best_model = model()
                best_model.fit(train_x, train_y)

            predict_y = best_model.predict(test_x)
            result_train = (best_model.predict(train_x) == train_y).sum()
            result_test = (best_model.predict(test_x) == test_y).sum()
            out = open(model_dirs['STEP3_logs'], 'a')
            train_acc = float(result_train) / len(train_y)
            test_acc = float(result_test) / len(test_y)
            out.write('%s,%f,%f\n' % (brandcode, train_acc, test_acc))
            # pdb.set_trace()
            out.close()
        if brandcode == startbrand:
            flag = True
        


