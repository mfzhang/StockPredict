# coding: utf-8

import cPickle, json, pdb, pickle, theano, sys, time, os
import numpy as np
import sklearn.decomposition
from sklearn.svm import SVR
from sklearn.svm import LinearSVC as SVC
from sklearn.decomposition import PCA
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



default_model_dir = 'experiment/Model'




############################
###  Setting parameters  ###
############################
dataset_type = 'test' # ['all' / 'chi2_selected']

params = {
    'experiment_type' : 'proposed',
    'STEP1' : {
        'reg_weight' : 1.,
        'model' : 'sae',
        'n_hidden' : 1000,
        'learning_rate' : 0.05,
        'corruption_level' : 0.5
    },
    'STEP3' : {
        'brandcode' : '0101'
    },
    'STEP4' : {
        'model' : 'sda',
        'corruption_levels' : [.3, .3, .3],
        'k' : 1,
        'hidden_layers_sizes' : [2500, 2500],
        'pretrain' : {
            'batch_size' : 50,
            'learning_rate' : 0.01,
            'epochs' : 100
        },
        'finetune' : {
            'batch_size' : 50,
            'learning_rate' : 0.01,
            'epochs' : 200
        }
    }
}


model_dirs = {}

#######################################
####   config: message in cosole   ####
#######################################

initial_msg = []
initial_msg.append('** どのステップから始めるかを入力して下さい。 **\n')
initial_msg.append('1: 圧縮モデル Sparse Auto-encoder / RBM の作成・訓練を行う')
initial_msg.append('2: 訓練された圧縮モデルを用い、複数記事を圧縮する')
initial_msg.append('3: 指定された銘柄の株価と記事データを組み合わせて銘柄の株価を予測する')
num_max = len(initial_msg)

labeltype_msg = []
labeltype_msg.append('以下から正解ラベルの形式を選択して下さい．\n')
labeltype_msg.append('1 : 回帰 : (終値 - 始値) / 終値')
labeltype_msg.append('2 : 回帰 : 翌日MACD - 当日MACD')
labeltype_msg.append('3 : 二値分類 : (終値 - 始値) <> 0')
labeltype_msg.append('4 : 二値分類 : 翌日MACD - 当日MACD <> 0')

x, y, z, i, l, m = '', '', '', '', '', ''

def msg_loop(stdscr):
    global x, y, z, i, l, m
    msg_head = ''
    curses.echo()

    while(1):
        curses.flushinp()
        stdscr.clear()
        msg = '\n'.join(initial_msg) + '\n\n'
        stdscr.addstr(msg)
        x = int(stdscr.getstr())
        if 0 < x < num_max:
            # curses.flushinp()
            # stdscr.clear()
            # msg = '**  ' + initial_msg[x] + '\n\n'
            # msg += 'このステップから後のいくつのステップを実行しますか？\n'
            # msg += '0: cancel, 1~: number of steps\n'
            # stdscr.addstr(msg)
            # y = int(stdscr.getstr())

            # if y > 0:
            curses.flushinp()
            stdscr.clear()
            msg = '**  ' + initial_msg[x] + '\n\n'
            msg = json.dumps(model_dirs, indent=2) + '\n'
            msg += '以下のファイルが上書きされる可能性があります。実行しますか？ [ y / n ]\n'
            for path in model_dirs.values():
                if os.path.exists(path):
                    msg += path + '\n'
            stdscr.addstr(msg)
            z = stdscr.getstr()
            if z == '' or z == 'y':
                break
            curses.flushinp()
            stdscr.clear()
        else:
            curses.flushinp()
            stdscr.clear()
            msg = '  ** 注 **     ' + str(num_max) + 'までの値を入力して下さい。\n'
            stdscr.addstr(msg)

    curses.flushinp()
    stdscr.clear()
    if x == 1:
        while True:
            msg = '**  ' + initial_msg[x] + '\n\n'
            msg += '以下を選択して下さい。\n'
            msg += '1: さいしょからはじめる、2: つづきからはじめる\n'
            stdscr.addstr(msg)
            i = int(stdscr.getstr())
            curses.flushinp()
            stdscr.clear()
            if i <= 2:
                break
            else:
                msg = '  ** 注 **     1 ~ 2 までの値を入力して下さい。\n'
                stdscr.addstr(msg)
    if x == 3:
        while True:
            msg = '**  ' + initial_msg[x] + '\n'
            msg += '利用するデータセットのタイプを選択して下さい．\n'
            msg += '1: proposed, 2: baseline\n'
            stdscr.addstr(msg)
            d = int(stdscr.getstr())
            if d == 1:
                params['experiment_type'] = 'proposed'
            else:
                params['experiment_type'] = 'baseline'
            stdscr.clear()
            
            msg = '**  ' + initial_msg[x] + '\n'
            msg += '**  ' + params['experiment_type'] + '\n'
            msg += '\n'.join(labeltype_msg) + '\n'
            stdscr.addstr(msg)
            l = int(stdscr.getstr())
            stdscr.clear()

            msg = '**  ' + initial_msg[x] + '\n'
            msg += '**  ' + params['experiment_type'] + '\n'
            msg += '**  ' + labeltype_msg[l] + '\n\n'
            msg += '以下から予測モデルに利用するモデルを選択して下さい．\n'
            msg += '1: SdA_regression, 2: DBN_regression, 3: SdA_RNN, 0: SVR\n'
            stdscr.addstr(msg)
            
            m = int(stdscr.getstr())
            curses.flushinp()
            stdscr.clear()
            if l <= 3 and m <= 4:
                break 
            else:
                print '  ** 注 **     適切な値を入力して下さい。\n'

def load_model(model_type='sae', input=None, params_dir=None):
    params = cPickle.load(open(params_dir))
    if model_type == 'rbm':
        model = RBM(input=input, params=params)
    else:
        model = SparseAutoencoder(input=input, params=params)
    return model

############################################################
#####          PHASE1: 複数記事の圧縮表現の獲得             #####
############################################################
###  STEP 1: Sparse Auto-encoder / RBM のモデルの作成・訓練  ###
############################################################

def build_CompressModel():
    print 'STEP 1 start...'
    dataset = Nikkei(dataset_type=params['experiment_type'], brandcode=params['STEP3']['brandcode'])
    # pdb.set_trace()
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    if params['STEP1']['model'] == 'rbm':
        model = RBM(input=x, n_visible=dataset.phase1_input_size, n_hidden=params['STEP1']['n_hidden'], reg_weight=params['STEP1']['reg_weight'], corruption_level=params['STEP1']['corruption_level'])
        train_rbm(input=x, model=model, dataset=dataset, learning_rate=params['STEP1']['learning_rate'], outdir=model_dirs['STEP1'])
    else:
        model = SparseAutoencoder(input=x, n_visible=dataset.phase1_input_size, n_hidden=params['STEP1']['n_hidden'], reg_weight=params['STEP1']['reg_weight'], corruption_level=params['STEP1']['corruption_level'])
        train_sae(input=x, model=model, dataset=dataset, learning_rate=params['STEP1']['learning_rate'], outdir=model_dirs['STEP1'])

def retrain_CompressModel():
    print 'STEP 1 start...'
    dataset = Nikkei(dataset_type=params['experiment_type'], brandcode=params['STEP3']['brandcode'])
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    if params['STEP1']['model'] == 'rbm':
        model = load_model(model_type='rbm', input=x, params_dir=model_dirs['STEP1'])
        train_rbm(input=x, model=model, dataset=dataset, learning_rate=params['STEP1']['learning_rate'], outdir=model_dirs['STEP1'])
    else:
        model = load_model(model_type='sae', input=x, params_dir=model_dirs['STEP1'])
        train_sae(input=x, model=model, dataset=dataset, learning_rate=params['STEP1']['learning_rate'], outdir=model_dirs['STEP1'])

######################################################################
###  STEP 2: 前のステップで訓練された圧縮モデルを用いた複数記事の圧縮表現の獲得  ###
######################################################################

def unify_kijis(dataset):
    print 'STEP 2 start...'
    if dataset == None:
        print 'dataset load...'
        dataset = Nikkei(dataset_type=params['experiment_type'], brandcode=params['STEP3']['brandcode'])
    # model = load_model(model_dirs['STEP1'])
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    ###########################################################
    model = load_model(input=x, params_dir=model_dirs['STEP1'], model_type=params['STEP1']['model'])
    # model = cPickle.load(open(model_dirs['STEP1']))
    ###########################################################
    dataset.unify_kijis(model, params['STEP1']['model'], params['experiment_type'])
    out = open(model_dirs['STEP2'], 'w')
    out.write(cPickle.dumps(dataset))
    return dataset

######################################################################
#####                 PHASE2: 各銘柄の株価の予測                     #####
######################################################################
###  STEP 3: 指定された銘柄の株価と記事データを組み合わせ，銘柄の株価を予測する  ###
######################################################################

def reguralize_data(dataset, brandcodes):
    for datatype in ['train', 'valid', 'test']:
        # pdb.set_trace()
        # dataset.phase2[datatype]['x'] = ((dataset.phase2[datatype]['x'] / dataset.phase2[datatype]['x'].max(axis=0)) ** 2)
        dataset.phase2[datatype]['x'] = ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0)) ** 2) / ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0) + 0.001) ** 2).max(axis=0)
        for brandcode in brandcodes:
            dataset.phase2[datatype][brandcode] /= dataset.phase2[datatype][brandcode].max()
        if theano.config.floatX == 'float32':
            print 'cast to 32bit matrix'
            dataset.phase2[datatype]['x'] = dataset.phase2[datatype]['x'].astype(np.float32)
            for brandcode in brandcodes:
                dataset.phase2[datatype][brandcode] = dataset.phase2[datatype][brandcode].astype(np.float32)

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


def predict(dataset, model, brandcodes=['0101'], label_type=1, model_type=1):
    print 'STEP 3 start...'
    if dataset == None:
        if params['experiment_type'] == 'baseline':
            print 'start to load baseline dataset...'
            dataset = cPickle.load(open(default_model_dir + '/STEP2/baseline_original'))
            print 'start to unify stockprice...'
            dataset.unify_stockprices(dataset=dataset.baseline_original, brandcodes=brandcodes, dataset_type=params['experiment_type'], label_type=label_type)
        else:
            print 'start to load proposed dataset...'
            dataset = cPickle.load(open(model_dirs['STEP2']))
            print 'start to unify stockprice...'
            dataset.unify_stockprices(dataset=dataset.unified, brandcodes=brandcodes, dataset_type=params['experiment_type'], label_type=label_type)
    
    
    
    if params['experiment_type'] != 'baseline':
        reguralize_data(dataset, brandcodes)
    change_brand(dataset, '0101')

    if model_type == 0:
        def transformY(data_y):
            y = []
            for data in data_y:
                y.append(data[0])
            return np.array(y)
        train_x = dataset.phase2['train']['x']
        train_x = np.append(train_x, dataset.phase2['valid']['x'], 0)
        test_x = dataset.phase2['test']['x']
        while(1):
            
            if params['experiment_type'] == 'baseline':
                train_x_original = train_x
                test_x_original = test_x
                pca = PCA(n_components=1000)
                pca.fit(train_x_original)
                train_x = pca.transform(train_x_original)
                test_x = pca.transform(test_x_original)
                    
            train_y = transformY(dataset.phase2['train']['y'])
            train_y = np.append(train_y, transformY(dataset.phase2['valid']['y']), 0)
            test_y = transformY(dataset.phase2['test']['y'])

            
            if label_type < 3:
                tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10**i for i in range(-4,0)], 'C': [10**i for i in range(0,4)]}]
                gscv = GridSearchCV(SVR(), tuned_parameters, cv=5, scoring="mean_squared_error", n_jobs=10)
            else:
                print 'classification'
                tuned_parameters = [{'C': [10**i for i in range(0,4)]}]
                gscv = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=-1)
            gscv.fit(train_x, train_y)
            best_svr = gscv.best_estimator_
            predict_y = best_svr.predict(test_x)
            print (predict_y == test_y).sum()
            pdb.set_trace()

    else:
        pretrain_params = {
            'dataset' : dataset, 
            'hidden_layers_sizes' : params['STEP4']['hidden_layers_sizes'],
            'pretrain_lr' : params['STEP4']['pretrain']['learning_rate'],
            'pretrain_batch_size' : params['STEP4']['pretrain']['batch_size'],
            'pretrain_epochs' : params['STEP4']['pretrain']['epochs'],
            'corruption_levels' : params['STEP4']['corruption_levels'],
            'k' : params['STEP4']['k']
        }
        pretrain_model = model.pretrain(pretrain_params)
        pre_params = get_model_params(pretrain_model)
        while(1):
            finetune_params = {
                'dataset' : dataset,
                'model' : pretrain_model,
                'finetune_lr' : params['STEP4']['finetune']['learning_rate'],
                'finetune_batch_size' : params['STEP4']['finetune']['batch_size'],
                'finetune_epochs' : params['STEP4']['finetune']['epochs']
            }
            finetune_model, best_validation_loss, test_score, best_epoch = model.finetune(finetune_params)
            pdb.set_trace()
            set_model_params(pretrain_model, pre_params)


##############
###  Main  ###
##############

if __name__ == '__main__':
   
    

    if len(sys.argv) > 5:
        params['STEP1']['n_hidden'] = int(sys.argv[1])
        params['STEP1']['learning_rate'] = float(sys.argv[2])
        params['STEP1']['reg_weight'] = float(sys.argv[3])
        params['STEP1']['corruption_level'] = float(sys.argv[4])
        params['STEP1']['model'] = sys.argv[5]
    else:
        print sys.argv
        print '引数が足りません．'
        print '引数: n_hidden learning_rate reg_weight model'
        sys.exit()

    model_dirs = {
        'STEP1' : '%s/%s/h%d_lr%s_b%s_c%s.%s' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['corruption_level']), params['STEP1']['model']),
        'STEP2' : '%s/%s/h%d_lr%s_b%s_c%s.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['corruption_level']), params['STEP1']['model']),
        'STEP3' : '%s/%s/h%d_lr%s_b%s_c%s.%s' % (default_model_dir, 'STEP3', params['STEP1']['n_hidden'], str(params['STEP1']['learning_rate']), str(params['STEP1']['reg_weight']), str(params['STEP1']['corruption_level']), params['STEP1']['model']),
        # 'STEP4' : '%s/%s/%sh%d_lr%.2f_b%s.%s' % (default_model_dir, 'STEP4', params['STEP3']['brandcode'], params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], str(params['STEP1']['reg_weight']), params['STEP1']['model']),
    }
    
    # print params

    curses.wrapper(msg_loop)
    sys.stdout = os.fdopen(0, 'w', 0)

    print initial_msg[x]

############################################################
#####          PHASE1: 複数記事の圧縮表現の獲得             #####
############################################################
###  STEP 1: Sparse Auto-encoder / RBM のモデルの作成・訓練  ###
############################################################

    dataset = None
    if x == 1:
        if i == 1:
            build_CompressModel()
        elif i == 2:
            retrain_CompressModel()
            
        
######################################################################
###  STEP 2: 前のステップで訓練された圧縮モデルを用いた複数記事の圧縮表現の獲得  ###
######################################################################

    if x == 2:
        dataset = unify_kijis(dataset)

######################################################################
#####                 PHASE2: 各銘柄の株価の予測                     #####
######################################################################
###  STEP 3: 指定された銘柄の株価と記事データを組み合わせ，銘柄の株価を予測する  ###
######################################################################


    if x == 3:
        model = ''
        print labeltype_msg[l]
        if m == 1:
            print 'start SdA_regression'
            model = SdA_regression
        elif m == 2:
            print 'start DBN_regression'
            model = DBN_regression
        elif m == 0:
            print 'start SVR'
            model = SVR
        brandcodes = ['0101', '7203', '6758', '6502', '7201', '6501', '6702', '6753', '8058', '8031', '7751']
        predict(dataset, model, brandcodes=brandcodes, label_type=l, model_type=m)



