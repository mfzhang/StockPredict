# coding: utf-8

import cPickle, json, pdb, pickle, theano, sys, numpy
import os.path
import theano.tensor as T
from dataset.Nikkei import Nikkei
from experiment.CompressSparseVector.SparseAutoencoder import SparseAutoencoder, train_sae
from experiment.CompressSparseVector.RBM import RBM, train_rbm
from experiment.PredictPrices.SdA_theano import SdA, train_SdA
# theano.config.floatX = 'float32'

default_model_dir = '/home/fujikawa/StockPredict/src/deeplearning/experiment/Model'




############################
###  Setting parameters  ###
############################
dataset_type = 'test' # ['all' / 'chi2_selected']

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

######################################################
#####      PHASE1: 複数記事の圧縮表現の獲得           #####
######################################################
###  STEP 1: Sparse Auto-encoder のモデルの作成・訓練  ###
######################################################

def build_CompressModel():
    print 'STEP 1 start...'
    dataset = Nikkei(dataset_type=params['dataset_type'], brandcode=params['STEP3']['brandcode'])
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    if params['STEP1']['model'] == 'rbm':
        model = RBM(input=x, n_visible=dataset.phase1_input_size, n_hidden=params['STEP1']['n_hidden'])
        train_rbm(input=x, model=model, dataset=dataset, learning_rate=params['STEP1']['learning_rate'], outdir=model_dirs['STEP1'])
    else:
        model = SparseAutoencoder(input=x, n_visible=dataset.phase1_input_size, n_hidden=params['STEP1']['n_hidden'], beta=params['STEP1']['beta'])
        train_sae(input=x, model=model, dataset=dataset, learning_rate=params['STEP1']['learning_rate'], outdir=model_dirs['STEP1'])

def retrain_CompressModel():
    print 'STEP 1 start...'
    dataset = Nikkei(dataset_type=params['dataset_type'], brandcode=params['STEP3']['brandcode'])
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    if params['STEP1']['model'] == 'rbm':
        model = RBM(input=x, n_visible=dataset.phase1_input_size, n_hidden=params['STEP1']['n_hidden'])
        train_rbm(input=x, model=model, dataset=dataset, learning_rate=params['STEP1']['learning_rate'], outdir=model_dirs['STEP1'])
    else:
        model = load_model(input=x, params_dir=model_dirs['STEP1'])
        train_sae(input=x, model=model, dataset=dataset, learning_rate=params['STEP1']['learning_rate'], outdir=model_dirs['STEP1'])

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

##########################################
###  STEP 4: 指定された銘柄の株価を予測する  ###
##########################################

def reguralize_data(dataset):
    for datatype in ['train', 'valid', 'test']:
        dataset.phase2[datatype]['x'] = ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0)) ** 2) / ((dataset.phase2[datatype]['x'] - dataset.phase2[datatype]['x'].min(axis=0)) ** 2).max(axis=0)
        if theano.config.floatX == 'float32':
            print 'cast to 32bit matrix'
            dataset.phase2[datatype]['x'] = dataset.phase2[datatype]['x'].astype(numpy.float32)
            dataset.phase2[datatype]['y'] = dataset.phase2[datatype]['y'].astype(numpy.float32)
def predict(dataset):
    print 'STEP 4 start...'
    if dataset == None:
        print 'dataset load...'
        dataset = cPickle.load(open(model_dirs['STEP3']))
    pdb.set_trace()
    reguralize_data(dataset)
    train_SdA(dataset=dataset, 
        hidden_layers_sizes=params['STEP4']['hidden_layers_sizes'],
        corruption_levels=params['STEP4']['corruption_levels'],
        pretrain_lr=params['STEP4']['pretrain']['learning_rate'],
        pretrain_batch_size=params['STEP4']['pretrain']['batch_size'],
        pretrain_epochs=params['STEP4']['pretrain']['epochs'],
        finetune_lr=params['STEP4']['finetune']['learning_rate'],
        finetune_batch_size=params['STEP4']['finetune']['batch_size'],
        finetune_epochs=params['STEP4']['finetune']['epochs']
    )



##############
###  Main  ###
##############

if __name__ == '__main__':
   
    msg = []
    msg.append('** どのステップから始めるかを入力して下さい。 **')
    msg.append('1: Sparse Auto-encoder のモデルの作成・訓練を行う')
    msg.append('2: 訓練されたSparse Auto-encoderを用い、複数記事を圧縮する')
    msg.append('3: 指定された銘柄の株価と記事データを組み合わせる')
    msg.append('4: 指定された銘柄の株価を予測する')

    num_max = len(msg)

    if len(sys.argv) > 1:
        params['STEP1']['n_hidden'] = int(sys.argv[1])
        params['STEP1']['learning_rate'] = float(sys.argv[2])
        params['STEP1']['beta'] = float(sys.argv[3])
    else:
        print sys.argv
        print '記事圧縮フェーズの隠れ層と学習率を入力して下さい。'
        sys.exit()
    model_dirs = {
        'STEP1' : '%s/%s/h%d_lr%f_b%f.%s.params' % (default_model_dir, 'STEP1', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP2' : '%s/%s/h%d_lr%f_b%f.%s' % (default_model_dir, 'STEP2', params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP3' : '%s/%s/%sh%d_lr%f_b%f.%s' % (default_model_dir, 'STEP3', params['STEP3']['brandcode'], params['STEP1']['n_hidden'], params['STEP1']['learning_rate'], params['STEP1']['beta'], params['STEP1']['model']),
        'STEP4' : '%s/%s/%s' % (default_model_dir, 'STEP4', 'sda.pkl')
    }
    print params
    while True:
        for m in msg:
            print m
        x = input()
        if 0 < x < num_max:
            print '**  ' + msg[x]
            print 'このステップから後のいくつのステップを実行しますか？'
            print '0: cancel, 1~: number of steps'
            y = input()
            if y > 0:
                print json.dumps(model_dirs, indent=2)
                print '以下のファイルが上書きされる可能性があります。実行しますか？ [ y / n ]'
                for path in model_dirs.values():
                    if os.path.exists(path):
                        print path
                z = raw_input()
                if z == '' or z == 'y':
                    break
        else:
            print '  ** 注 **     ' + str(num_max) + 'までの値を入力して下さい。\n'

    dataset = None
    if x <= 1 and y > 0:
        y -= 1
        while True:
            print '以下を選択して下さい。'
            print '1: さいしょからはじめる、2: つづきからはじめる'
            i = input()
            if i == 1:
                build_CompressModel()
                break
            elif i == 2:
                retrain_CompressModel()
                break
            else:
                print '  ** 注 **     1 ~ 2 までの値を入力して下さい。\n'

        
        
    if x <= 2 and y > 0:
        y -= 1
        dataset = unify_kijis(dataset)

    if x <= 3 and y > 0:
        y -= 1
        dataset = unify_stockprices(dataset)

    if x <= 4 and y > 0:
        y -= 1
        predict(dataset)



