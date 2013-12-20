# coding: utf-8
import json, codecs, numpy, cPickle, gzip, utils, sys, nikkei225, datetime
bow_dic = json.load(codecs.open('/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/BOW/dat/bow-dic/chi2-1000.dic', 'r', 'utf-8'))
resdir = '/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/FeatureVectors/chi2.feature'
#hadoop_res = open('sample/bow-feature')
hadoop_res = open('/home/fujikawa/StockPredict/res-int/Nikkei/DataForDL/BOW/dat/bow-feature/chi2-1000.bowfeature')
lines = hadoop_res.readlines()
# vectors = {"brand" : {}, "category" : {}}
vectors = {}

def checkEnoughData(vec):
	flag = True
	threshold = 10
	for element in vec:
		print str(len(element[0])) + ', ' + str(len(element[1]) )
		if (len(element[0]) < threshold) or (len(element[1]) < threshold):
			flag = False
	return flag

print 'start to make vectors'

a = 0
brandlist = nikkei225.getNikkei225()
for t in ['brand']:
	for bc in brandlist:
		a += 1
		n = 0
		utils.Progress.print_progress(a, len(brandlist) * 2)

		c = 0
		for line in lines:
			# n += 1
			# if utils.Progress.get_progress(n, len(lines) * 2):
			# 	print '\t',
			# 	utils.Progress.print_progress(n, len(lines))
			type, brand_code, date, rate, wordid_list = line.strip().split('\t')
			year, month, day = date.strip().split('-')
			dt = datetime.date(int(year), int(month), int(day))
			if (type == t and brand_code == str(bc)) or brand_code == '0101':
				c += 1
				# wordid_list = wordid_list.split(',')
				# vector = [0 for i in range(len(bow_dic))]
				#vector = numpy.zeros(len(bow_dic[type][bc]))
				# for id in wordid_list:
				# 	vector[int(id)] = 1
				year = int(date.split('-')[0])
				dataType = 0
				if year == 2008:
					dataType = 2
				elif year >= 2006:
					dataType = 1

				if brand_code not in vectors:
					vectors[brand_code] = {}
				if year not in vectors[brand_code]:
					vectors[brand_code][int(year)] = {}
				vectors[brand_code][int(year)][dt] = map(int, wordid_list.split(','))
				# vectors[dataType][0].append(wordid_list)
				#vectors[dataType][0].append(vector)
				# vectors[dataType][1].append([float(rate)])
				line = hadoop_res.readline()
		# if checkEnoughData(vectors):
		print str(bc) + ' :: ' + str(c)
		# 	outdir = resdir + type + '/' + bc
		# 	f_out = gzip.open(outdir, 'w')
		# 	f_out.write(cPickle.dumps(vectors))
		# 	f_out.close()

print 'writing files...'
f_out = open(resdir, 'w')
f_out.write(cPickle.dumps(vectors))
f_out.close()

