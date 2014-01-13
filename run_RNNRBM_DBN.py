# coding: utf-8
from dataset.Nikkei import Nikkei
from yoshihara.PredictPrices.RNNRBM_DBN import RNNRBM_DBN,train_rnnrbm_dbn

params = {
    'dataset_type' : 'chi2_selected',
    'STEP3' : {
        'brandcode' : '0101'
    },
    'STEP4' : {
            'hidden_layers_size' : [1000,1000],
            'hidden_recurrent' : 10,
            'pretrain' : {
            'batch_size' :100, 
            'learning_rate' : 0.001,
            'epochs' : 20
        },
        'finetune' : {
            'batch_size' :30,
            'learning_rate' : 1.8,
            'epochs' :20
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
    train_rnnrbm_dbn(dataset=dataset,
        hidden_layers_sizes=params['STEP4']['hidden_layers_size'],
        hidden_recurrent = params['STEP4']['hidden_recurrent'],
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
