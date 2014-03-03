# coding: utf-8
import numpy as np
from progressbar import ProgressBar
import theano
import theano.tensor as T
import scipy.sparse as sp
import json, codecs, cPickle, gzip, datetime, pdb, sys

##  日ごと / 記事ごとに出現する単語のIDをまとめたデータセットのディレクトリ

# wordidset_all = "dataset/dataset/chi2-unified.wordidset"
wordidset_sentence = "dataset/dataset/chi2-unified-sentence-10000.wordidset"
wordidset_article = "dataset/dataset/chi2-unified-article-10000.wordidset"

# wordidset_all = "/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/FeatureVectors/chi2-unified.wordidset"
# wordidset_all = "/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/FeatureVectors/chi2-unified-sentence.wordidset"
wordidset_chi2_selected = "dataset/dataset/chi2.wordidset"

##  株価 / 辞書データに関するディレクトリ  

pricelistdir = 'dataset/dataset/pricelist.pkl'
dicdir = 'dataset/dataset/chi2-result-unified_10000.dic'

# pricelistdir = '/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/FeatureVectors/StockPrice/pricelist.pkl'
# dicdir = '/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/BOW/dat/bow-dic/chi2-result-unified.dic'

class Nikkei():
    def __init__(self, type="theano_sparse", dataset_type='chi2_selected', brandcode ='0101'):
        """
        日経新聞・株価に関するデータを構造化して保持するクラス
        :param type : ["theano_sparse" / "theano_dense" / "scipy_sparse"]
        :param dataset_type : "chi2_selected" : 銘柄ごとにカイ二乗検定を行い、上位1000語を辞書にしたもの。日ごとに集計しているため、圧縮表現の獲得の必要なし。
                              "all" : 提案手法で利用する、記事ごとに集計したもの。
                                               | vocaburary | group by |    filtering with    |
                              "chi2_selected"  |    1000    |    days  | brand name / synomym |
                              "all"            |    5000    |   kijis  |        nothing       |
        :param phase1 : 圧縮表現獲得部分で使うデータセット
        :param phase2 : 株価予測で使うデータセット

        """

        ##  データセットの読み込み
        
        print dataset_type
        if dataset_type == 'chi2_selected':
            datasetdir = wordidset_chi2_selected
            n_dic = 1000
        elif dataset_type == 'article':
            n_dic = len(json.load(open(dicdir)))
            datasetdir = wordidset_article
        elif dataset_type == 'sentence':
            n_dic = len(json.load(open(dicdir)))
            datasetdir = wordidset_sentence
        else:
            sys.exit()
            datasetdir = wordidset_all
            n_dic = len(json.load(open(dicdir)))

        self.raw_data = cPickle.load(open(datasetdir))
        print n_dic
        self.type = type
        self.phase1_input_size = n_dic
        self.phase2_input_size = 1000
        dataset = self.raw_data
        if dataset_type == 'chi2_selected':
            dataset = self.raw_data[brandcode]

        print 'finish loading data...'

        ##  train, valid, testの割り振り設定
        self.years = {}
        self.years['train'] = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006]
        self.years['valid'] = [2007]
        self.years['test'] = [2008]

        print 'year change'
        # self.years['train'] = [1999, 2000, 2001, 2002, 2003, 2004]
        # self.years['valid'] = [2005, 2006]
        # self.years['test'] = [2007, 2008]

        ################################################
        ###  PHASE1: 複数記事の圧縮表現獲得部分のデータ設定  ###
        ################################################
        self.trainset, self.validset, self.testset = [], [], []
        self.phase1 = {}
        self.unified_max = {}
        self.unified_mean = {}
        self.baseline = {}
        self.baseline_original = {}
        if dataset_type != 'chi2_selected':
            self.prepare_phase1_data(dataset)

        ###################################################
        ###  PHASE2: 株価予測部分のデータを格納する変数の初期化  ###
        ###################################################
        self.phase2 = {}
        self.phase2['train'] = {'x': []}
        self.phase2['valid'] = {'x': []}
        self.phase2['test'] = {'x': []}
        
    def prepare_phase1_data(self, dataset):
        """
        PHASE1の実験に用いるデータの準備

        """
        for year in self.years['train']:
            self.trainset.extend(self._expandArray(dataset[year].values()))
        for year in self.years['valid']:
            self.validset.extend(self._expandArray(dataset[year].values()))
        for year in self.years['test']:
            self.testset.extend(self._expandArray(dataset[year].values()))
        self.phase1['train'] = self.get_data(self.trainset, type=self.type)
        self.phase1['valid'] = self.get_data(self.validset, type=self.type)
        self.phase1['test'] = self.get_data(self.testset, type=self.type)

    def unify_stockprices(self, dataset=None, brandcodes=['0101'], dataset_type='chi2_selected', label_type=1, y_type=0, y_force_list=False):
        """
        ########                      STEP 3                           #########
        指定された銘柄の、記事に関する素性と株価を日付で結びつけ、self.phase2へ保存する
        :param brandcode (string) : 銘柄コード（'0101' : 日経平均、'7203' : トヨタなど）
        :param label_type (int) : 正解ラベルの種類
                                  1 : 回帰 : (終値 - 始値) / 終値
                                  2 : 回帰 : 翌日MACD - 当日MACD
                                  3 : 二値分類 : (終値 - 始値) <> 0
                                  4 : 二値分類 : 翌日MACD - 当日MACD <> 0
        """
        count_zero = 0
        count_one = 0
        count_two = 0
        print 'year change'
        self.years['train'] = [1999, 2000, 2001, 2002, 2003, 2004]
        self.years['valid'] = [2005, 2006]
        self.years['test'] = [2007, 2008]
        ng_brands = []
        previous_year, previous_date = None, None
        self.pricelist = cPickle.load(open(pricelistdir))
        for datatype in ['train', 'valid', 'test']:
            for year in self.years[datatype]:
                if year in self.pricelist['0101']:
                    pricelist_datelist = self.pricelist['0101'][year].keys()
                    pricelist_datelist.sort()
                    for date in pricelist_datelist:
                        ## 株価とニュース記事共にある日のみを対象とする
                        if year in dataset and date in dataset[year]:
                            self.phase2[datatype]['x'].append(dataset[year][date])
                            for brandcode in brandcodes:
                            ##!! TODO : 当日の終値と始値の差を予測するタスクにしているが、MACDを利用したり、もう少し長期的に見たりいろいろできるので要検討
                                label = 0
                                ## 回帰 : (終値 - 始値) / 終値
                                if (year not in self.pricelist[brandcode]) or (date not in self.pricelist[brandcode][year]):
                                    if brandcode not in ng_brands:
                                        print '!!!!!!!!  ' + brandcode + ' : ' + str(date)
                                        ng_brands.append(brandcode)
                                    # break
                                    # break
                                else:
                                    if label_type == 1:
                                        if self.pricelist[brandcode][year][date]['closing_price'] == 0:
                                            label = 0
                                        else:
                                            label = float(self.pricelist[brandcode][year][date]['closing_price'] - self.pricelist[brandcode][year][date]['opening_price']) / self.pricelist[brandcode][year][date]['closing_price']
                                    ## 回帰 : 翌日MACD - 当日MACD
                                    elif label_type == 2:
                                        label = self.pricelist[brandcode][year][date]['macd_tomorrow'] - self.pricelist[brandcode][year][date]['macd']
                                    ## 二値分類 : (終値 - 始値) <> 0
                                    elif label_type == 3:
                                        label = int((self.pricelist[brandcode][year][date]['closing_price'] - self.pricelist[brandcode][year][date]['opening_price']) > 0)
                                    ## 二値分類 : 翌日MACD - 当日MACD <> 0
                                    elif label_type == 4:
                                        diff = self.pricelist[brandcode][year][date]['macd_tomorrow'] - self.pricelist[brandcode][year][date]['macd']
                                        label = int((self.pricelist[brandcode][year][date]['macd_tomorrow'] - self.pricelist[brandcode][year][date]['macd']) > 0)
                                    elif label_type == 5:
                                        if previous_year == None:
                                            label = 0
                                        else:
                                            previous_diff = self.pricelist[brandcode][previous_year][previous_date]['macd_tomorrow'] - self.pricelist[brandcode][previous_year][previous_date]['macd']
                                            diff = self.pricelist[brandcode][year][date]['macd_tomorrow'] - self.pricelist[brandcode][year][date]['macd']
                                            label = int((previous_diff > 0) != (diff > 0))
                                        
                                        

                                    ## y のデータを格納する配列の準備
                                    if brandcode not in self.phase2[datatype]:
                                        self.phase2[datatype][brandcode] = []

                                    # self.phase2[datatype][brandcode].append([label])
                                    if y_type == 0 or y_force_list == True:
                                        self.phase2[datatype][brandcode].append([label])
                                    else:
                                        self.phase2[datatype][brandcode].append(label)
                            previous_year = year
                            previous_date = date
            # pdb.set_trace()

            if dataset_type == 'chi2_selected':
                self.phase2[datatype]['x'] = self.get_numpy_dense_design(self.phase2[datatype]['x'])
            else:
                self.phase2[datatype]['x'] = np.asarray(self.phase2[datatype]['x'], dtype=theano.config.floatX)

            for brandcode in brandcodes:
                try:
                    self.phase2[datatype][brandcode] = np.asarray(self.phase2[datatype][brandcode], dtype=theano.config.floatX)
                except:
                    pass


# theano、scipyなど、様々なデータ形式へ変換して取得
    def get_data(self, data, type=None):
        """
        各種データタイプへの変換
        """
        if type == None:
            type = self.type
        if type =='theano_dense':
            return self.get_theano_design(self.get_numpy_dense_design(data))
        elif type == 'theano_sparse':
            return self.get_theano_design(self.get_scipy_sparse_design(data, dtype=theano.config.floatX))
        elif type == 'scipy_sparse':
            return self.get_scipy_sparse_design(data)
        else:
            print 'Invalid "type"'
            sys.exit()

    def get_batch_design(self, index, batch_size, dataset, type=None, isInt=False):
        """
        バッチの獲得

        """
        if type == None:
            type = self.type

        if type == 'theano_sparse':
            return theano.sparse.dense_from_sparse( dataset[index * batch_size:  (index + 1) * batch_size] )
        elif type == 'numpy_dense':
            n_batches = len(dataset['x']) / batch_size
            batch_design_data_x = []
            batch_design_data_y = []
            if isInt:
                for i in range(n_batches):
                    batch_design_data_x_tmp = []
                    batch_design_data_y_tmp = []
                    for j in range(batch_size):
                        batch_design_data_x_tmp.append(dataset['x'][batch_size * i + j])
                        batch_design_data_y_tmp.append(dataset['y'][batch_size * i + j])
                    batch_design_data_x.append(batch_design_data_x_tmp)
                    batch_design_data_y.append(batch_design_data_y_tmp)
                    # batch_design_data_x.append(dataset['x'][batch_size * i: batch_size * (i + 1)])
                    # batch_design_data_y.append(np.asarray(dataset['y'][batch_size * i: batch_size * (i + 1)], dtype=np.int32))
            else:
                for i in range(n_batches):
                    batch_design_data_x.append(dataset['x'][batch_size * i: batch_size * (i + 1)])
                    batch_design_data_y.append(dataset['y'][batch_size * i: batch_size * (i + 1)])

            return batch_design_data_x, batch_design_data_y
        else:
            print 'Cannot get batch design'
            sys.exit()

    def unify_kijis(self, model, model_type, experiment_type):
        """
        ######                STEP 2                   #######
        記事を日ごとに統合
        :param model (SparseAutoencoder / RBM)
        :param model_type (string) : [rbm / sae]

        """
        
        self.phase2_input_size = model.n_hidden
        for year in self.raw_data.keys():
            print year
            if year not in self.unified_max:
                self.unified_max[year] = {}
                self.unified_mean[year] = {}
                self.baseline[year] = {}
                # self.baseline_original[year] = {}
            bar = ProgressBar(maxval=len(self.raw_data[year].keys())).start()
            for i, date in enumerate(self.raw_data[year].keys()):
                bar.update(i)
                vectors = self.get_numpy_dense_design(self.raw_data[year][date])
                
                vectors_baseline = np.max(vectors, axis=0)
                self.baseline[year][date] = vectors_baseline

                if experiment_type != 'baseline':

                    daily_vector_maxpool = model.get_maxpool(vectors)
                    daily_vector_meanpool = model.get_meanpool(vectors)  
                    self.unified_max[year][date] = daily_vector_maxpool
                    self.unified_mean[year][date] = daily_vector_meanpool

        if experiment_type == 'baseline':
            self.raw_data = None
            self.trainset, self.validset, self.testset = [], [], []
            self.phase1 = {}

    def get_theano_design(self, array):
        return theano.shared(array)

    def get_numpy_dense_design(self, idlists):
        n_vectors = len(idlists)
        # print n_vectors, n_dic
        vectors = np.zeros([n_vectors, self.phase1_input_size], dtype=theano.config.floatX)
        for i, idlist in enumerate(idlists):
            # vector = [0 for i in xrange(n_dic)]
            # vector = np.zeros(n_dic)
            for id in idlist:
                vectors[i][int(id)] = 1
            # vectors.append(vector)
        return vectors

    def get_scipy_sparse_design(self, idlists, dtype=None):
        data = []
        row = []
        col = []
        for i, idlist in enumerate(idlists):
            for id in idlist:
                row.append(i)
                col.append(id)
                data.append(1)
        # pdb.set_trace()
        return sp.csr_matrix((data, (row, col)), shape=(len(idlists), self.phase1_input_size), dtype=dtype)

    def _expandArray(self, array):
        ex_array = []
        for a1 in array:
            for a2 in a1:
                ex_array.append(a2)
        return ex_array

   

if __name__ == '__main__':
    pass
    # nikkei = cPickle.load(open('/home/fujikawa/StockPredict/src/deeplearning/experiment/Model/unified_numpy'))
    # dataset = Nikkei()
    # modeldir = '/home/fujikawa/StockPredict/src/deeplearning/experiment/Model/sae.pkl'
    # model = cPickle.load(open(modeldir))
    # dataset.unify_kijis(model)
    # out = open('/home/fujikawa/StockPredict/src/deeplearning/experiment/Model/unified_numpy.pkl', 'w')
    # out.write(cPickle.dumps(dataset))
    # dataset.unify_stockprices()
    # out = open('/home/fujikawa/StockPredict/src/deeplearning/experiment/Model/unified_stockprice.pkl', 'w')
    # out.write(cPickle.dumps(dataset))

    # pdb.set_trace()
    
