# coding: utf-8
from dataset.Nikkei import Nikkei
from yoshihara.PredictPrices.DBN import DBN, train_DBN
from dataset.xor import xor
params = {
    'dataset_type' : 'chi2_selected',
    'STEP3' : {
        'brandcode' : '0101'
    },
    'STEP4' : {
        'hidden_layers_sizes' : [2500],
        'pretrain' : {
            'batch_size' : 20,
            'learning_rate' : 0.01,
            'epochs' : 50 
        },
        'finetune' : {
            'batch_size' :10, 
            'learning_rate' : 1.25,
            'epochs' :50 
        }
    }
}

######################################################
###  STEP 3: 指定された銘柄の株価と記事データを組み合わせる  ###
######################################################

def unify_stockprices(dataset):
    print 'STEP 3 start...'
    dataset.unify_stockprices(dataset.raw_data[params['STEP3']['brandcode']])

##########################################
###  STEP 4: 指定された銘柄の株価を予測する  ###
##########################################

def predict(dataset):
    print 'STEP 4 start...'
    train_DBN(dataset=dataset, 
        hidden_layers_sizes=params['STEP4']['hidden_layers_sizes'],
        pretrain_lr=params['STEP4']['pretrain']['learning_rate'],
        pretrain_batch_size=params['STEP4']['pretrain']['batch_size'],
        pretrain_epochs=params['STEP4']['pretrain']['epochs'],
        finetune_lr=params['STEP4']['finetune']['learning_rate'],
        finetune_batch_size=params['STEP4']['finetune']['batch_size'],
        finetune_epochs=params['STEP4']['finetune']['epochs']
    )


if __name__ == '__main__':
    dataset = Nikkei(dataset_type=params['dataset_type'], brandcode=params['STEP3']['brandcode'])
    unify_stockprices(dataset)
    predict(dataset)
