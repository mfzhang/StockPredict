# coding: utf-8
import json, codecs, numpy, cPickle, gzip, utils, datetime
bow_dic = json.load(codecs.open('/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/BOW/dat/bow-dic/chi2-result-unified.dic', 'r', 'utf-8'))
resdir = '/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/FeatureVectors/chi2-unified-sentence.wordidset'
#hadoop_res = open('sample/bow-feature')
hadoop_res = open('/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/BOW/dat/bow-feature/chi2-unified-sentence.bowfeature')
lines = hadoop_res.readlines()
vectors = {}

def checkEnoughData(vec):
	flag = True
	threshold = 5
	for element in vec:
		print str(len(element[0])) + ', ' + str(len(element[1]) )
		if (len(element[0]) < threshold) or (len(element[1]) < threshold):
			flag = False
	return flag

print 'start to make vectors'

n = 0
c = 0
for line in lines:
	# n += 1
	# if utils.Progress.get_progress(n, len(lines) * 2):
	# 	print '\t',
	# 	utils.Progress.print_progress(n, len(lines))
	date, kiji_id, sentence_num, wordid_list = line.strip().split('\t')
	year, month, day = date.strip().split('-')
	dt = datetime.date(int(year), int(month), int(day))
	if int(year) not in vectors:
		vectors[int(year)] = {}
	if dt not in vectors[int(year)]:
		vectors[int(year)][dt] = []
	if len(wordid_list.split(',')) > 1:
		vectors[int(year)][dt].append(map(int, wordid_list.split(',')))
	# line = hadoop_res.readline()
	
# import pdb; pdb.set_trace()
f_out = open(resdir, 'w')
f_out.write(cPickle.dumps(vectors))
f_out.close()
