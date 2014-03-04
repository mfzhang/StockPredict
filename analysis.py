import pandas
import sys, pdb
data = pandas.read_csv(sys.argv[1])
hyper_parameter = ['hidden_layers_sizes', 'batch_size_pretrain', 'learning_rate_pretrain', 'hidden_recurrent','epochs_pretrain', 'batch_size_finetune', 'learning_rate_finetune', 'epochs_finetune']
#print data.groupby(hyper_parameter).test_score.mean()
grouped = data.groupby(hyper_parameter)
mean = grouped.test_score.mean()
print mean.order()
