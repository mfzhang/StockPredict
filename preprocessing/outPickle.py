# coding: utf-8
import json, codecs, numpy, cPickle, gzip, utils, nikkei225
bow_dic = json.load(codecs.open('/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/BOW/dat/bow-dic/threshold_100.json', 'r', 'utf-8'))
resdir = '/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/FeatureVectors/red_100/'
#hadoop_res = open('sample/bow-feature')
hadoop_res = open('/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/BOW/dat/bow-feature_reduce')
lines = hadoop_res.readlines()
vectors = {"brand" : {}, "category" : {}}

def checkEnoughData(vec):
	flag = True
	threshold = 5
	for element in vec:
		print str(len(element[0])) + ', ' + str(len(element[1]) )
		if (len(element[0]) < threshold) or (len(element[1]) < threshold):
			flag = False
	return flag

print 'start to make vectors'

a = 0
brandlist = nikkei225.getNikkei225()
for t in ['brand', 'category']:
	for bc in brandlist:
		a += 1
		n = 0
		utils.Progress.print_progress(a, len(brandlist) * 2)
		vectors = [[], []], [[], []], [[], []] # train[train_x, train_y], valid[valid_x, valid_y], test[test_x, test_y]
		c = 0
		for line in lines:
			# n += 1
			# if utils.Progress.get_progress(n, len(lines) * 2):
			# 	print '\t',
			# 	utils.Progress.print_progress(n, len(lines))
			type, brand_code, date, rate, wordid_list = line.strip().split('\t')
			if type == t and brand_code == str(bc):
				c += 1
				wordid_list = wordid_list.split(',')
				# vector = [0 for i in range(len(bow_dic))]
				# vector = numpy.zeros(len(bow_dic))
				# for id in wordid_list:
				# 	vector[int(id)] = 1
				year = int(date.split('-')[0])
				dataType = 0
				if year == 2008:
					dataType = 2
				elif year >= 2006:
					dataType = 1
				vectors[dataType][0].append(wordid_list)
				vectors[dataType][1].append(float(rate))
				line = hadoop_res.readline()
		if checkEnoughData(vectors):
			print str(bc) + ' :: ' + str(c)
			outdir = resdir + type + '/' + bc
			f_out = gzip.open(outdir, 'w')
			f_out.write(cPickle.dumps(vectors))
			f_out.close()
