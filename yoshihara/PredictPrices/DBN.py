"""
"""
import cPickle
import gzip
import os
import sys
import time
import pdb
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

sys.path.append('../../tutorial')

from tutorial.LogisticRegression import LogisticRegression
from tutorial.HiddenLayer import HiddenLayer
from tutorial.rbm import RBM


class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=1, y_type=1,activation_function=None):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
	


        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        if y_type==0:
            self.y = T.matrix('y')  # the labels are presented as 1D vector
        else: 
            self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels

        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            def maxout(z = None):
                #g = theano.shared(numpy.zeros((hidden_layers_sizes[i],)),name='g',borrow=True)
                g = T.max(z[0:5])
                g = T.stack(g,T.max(z[5:10]))
                for index in xrange(hidden_layers_sizes[i]-10):
                    g = T.concatenate([g,[T.max(z[5*(index+2):5*(index+3)])]])
		return g
            
            if activation_function == "maxout":
                print "activation_funtion is maxout....."
                self.activation_function = maxout
 		sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=5*hidden_layers_sizes[i],
                                        activation=self.activation_function)
            else :
                print "activation_funtion is sigmoid....."
                self.activation_function = T.nnet.sigmoid
		sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=self.activation_function)
	    
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b,
                            y_type=y_type)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs, y_type=y_type)
        
        self.get_prediction = theano.function(
	    inputs = [self.x],
	    outputs = [self.logLayer.y_pred]
	    )

        self.get_py = theano.function(
	    inputs = [self.x],
	    outputs = [self.logLayer.p_y_given_x]
        )
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        #self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        if y_type == 0:
            self.finetune_cost = self.logLayer.squared_error(self.y)
        else:
            self.finetune_cost = self.logLayer.negative_log_likelihood(self.y) 

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(inputs=[index,
                            theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                    train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, dataset, batch_size, learning_rate, y_type):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        train_set_x, train_set_y = theano.shared(dataset.phase2['train']['x']), theano.shared(dataset.phase2['train']['y'])
        valid_set_x, valid_set_y = theano.shared(dataset.phase2['valid']['x']), theano.shared(dataset.phase2['valid']['y'])
        test_set_x, test_set_y = theano.shared(dataset.phase2['test']['x']), theano.shared(dataset.phase2['test']['y'])
        if not y_type==0:
            train_set_y = T.cast(train_set_y,'int32') 
            valid_set_y = T.cast(valid_set_y,'int32')
            test_set_y = T.cast(test_set_y,'int32')

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]},
	                    name='train')

        test_score_i = theano.function([index], self.errors,
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y: test_set_y[index * batch_size:
                                            (index + 1) * batch_size]},
		                name='test')

        valid_score_i = theano.function([index], self.errors,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]},
		                name='valid')

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def pretrain(pretrain_params, y_type):

    ############################
    ###  Setting parameters  ###
    ############################

    dataset = pretrain_params['dataset']
    hidden_layers_sizes = pretrain_params['hidden_layers_sizes']
    pretrain_lr = pretrain_params['pretrain_lr']
    pretrain_batch_size = pretrain_params['pretrain_batch_size']
    pretrain_epochs = pretrain_params['pretrain_epochs']
    k = pretrain_params['k']
    n_outs = pretrain_params['n_outs']
    ############################
    
    train_set_x, train_set_y = theano.shared(dataset.phase2['train']['x']), theano.shared(dataset.phase2['train']['y'])
    valid_set_x, valid_set_y = theano.shared(dataset.phase2['valid']['x']), theano.shared(dataset.phase2['valid']['y'])
    test_set_x, test_set_y = theano.shared(dataset.phase2['test']['x']), theano.shared(dataset.phase2['test']['y'])
    
    if not y_type == 0:
        train_set_y = T.cast(train_set_y,'int32') 
        valid_set_y = T.cast(valid_set_y,'int32')
        test_set_y = T.cast(test_set_y,'int32')
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / pretrain_batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    """
    model = DBN(numpy_rng=numpy_rng, n_ins=train_set_x.get_value().shape[1],
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=n_outs, y_type=y_type,activation_function="maxout")
    """
    model = DBN(numpy_rng=numpy_rng, n_ins=train_set_x.get_value().shape[1],
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=n_outs, y_type=y_type)
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = model.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=pretrain_batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(model.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretrain_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            msg = 'Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c))
            sys.stdout.write("\r%s" % msg)
            sys.stdout.flush()
        print

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = "", "", "", "", "", ""
    return model

def finetune(finetune_params, y_type):

    ############################
    ###  Setting parameters  ###
    ############################

    dataset = finetune_params['dataset']
    model = finetune_params['model']
    finetune_lr = finetune_params['finetune_lr']
    finetune_batch_size = finetune_params['finetune_batch_size']
    finetune_epochs = finetune_params['finetune_epochs']

    ############################

    train_set_x, train_set_y = theano.shared(dataset.phase2['train']['x']), theano.shared(dataset.phase2['train']['y'])
    valid_set_x, valid_set_y = theano.shared(dataset.phase2['valid']['x']), theano.shared(dataset.phase2['valid']['y'])
    test_set_x, test_set_y = theano.shared(dataset.phase2['test']['x']), theano.shared(dataset.phase2['test']['y'])

    if not y_type==0 :
        train_set_y = T.cast(train_set_y,'int32')
        valid_set_y = T.cast(valid_set_y,'int32')
        test_set_y = T.cast(test_set_y,'int32')
    ########################
    # FINETUNING THE MODEL #
    ########################

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / finetune_batch_size
    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = model.build_finetune_functions(
                dataset=dataset, batch_size=finetune_batch_size,
                learning_rate=finetune_lr, y_type=y_type)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 30 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_epoch = 0
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0


    while (epoch < finetune_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                msg = ('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                sys.stdout.write("\r%s" % msg)
                sys.stdout.flush()
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    best_epoch = epoch
                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    print
    print test_score
    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = "", "", "", "", "", ""
    return model, best_validation_loss, test_score, best_epoch

if __name__ == '__main__':
    pass
