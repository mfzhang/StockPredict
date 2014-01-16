# coding: utf-8

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


sys.path.append('/home/fujikawa/StockPredict/src/deeplearning/tutorial')

from tutorial.LogisticRegression import LogisticRegression
from tutorial.HiddenLayer import HiddenLayer
from tutorial.dA import dA


class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=1,
                 corruption_levels=[0.1, 0.1]):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.matrix('y')  # the labels are presented as 1D vector of
        # self.y = T.ivector('y') 

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input=self.sigmoid_layers[-1].output,
                         n_in=hidden_layers_sizes[-1], n_out=n_outs,
                         y_type=0
                         )

        self.get_prediction = theano.function(
            inputs=[self.x],
            outputs=[self.logLayer.y_pred]
            )
        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        # self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.finetune_cost = self.logLayer.squared_error(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)


    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, dataset, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
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
        # train_set_x, train_set_y = theano.shared(dataset.phase2['train']['x']), T.cast(theano.shared(dataset.phase2['train']['y']), 'int32')
        # valid_set_x, valid_set_y = theano.shared(dataset.phase2['valid']['x']), T.cast(theano.shared(dataset.phase2['valid']['y']), 'int32')
        # test_set_x, test_set_y = theano.shared(dataset.phase2['test']['x']), T.cast(theano.shared(dataset.phase2['test']['y']), 'int32')
 
        train_set_x, train_set_y = theano.shared(dataset.phase2['train']['x']), theano.shared(dataset.phase2['train']['y'])
        valid_set_x, valid_set_y = theano.shared(dataset.phase2['valid']['x']), theano.shared(dataset.phase2['valid']['y'])
        test_set_x, test_set_y = theano.shared(dataset.phase2['test']['x']), theano.shared(dataset.phase2['test']['y'])

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
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]},
              name='train')

        test_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]},
                      name='test')

        valid_score_i = theano.function([index], self.errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
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


def pretrain(pretrain_params):

    ############################
    ###  Setting parameters  ###
    ############################

    dataset = pretrain_params['dataset']
    hidden_layers_sizes = pretrain_params['hidden_layers_sizes']
    pretrain_lr = pretrain_params['pretrain_lr']
    pretrain_batch_size = pretrain_params['pretrain_batch_size']
    pretrain_epochs = pretrain_params['pretrain_epochs']
    corruption_levels = pretrain_params['corruption_levels']

    ############################

    train_set_x, train_set_y = theano.shared(dataset.phase2['train']['x']), theano.shared(dataset.phase2['train']['y'])
    valid_set_x, valid_set_y = theano.shared(dataset.phase2['valid']['x']), theano.shared(dataset.phase2['valid']['y'])
    test_set_x, test_set_y = theano.shared(dataset.phase2['test']['x']), theano.shared(dataset.phase2['test']['y'])

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= pretrain_batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    
    model = SdA(numpy_rng=numpy_rng, n_ins=train_set_x.get_value().shape[1],
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=1)

    #########################
    # PRETRAINING THE MODEL #
    #########################

    print '... getting the pretraining functions'
    pretraining_fns = model.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=pretrain_batch_size)
    # pdb.set_trace()
    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    # pdb.set_trace()
    for i in xrange(model.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretrain_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            msg = 'Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c))
            sys.stdout.write("\r%s" % msg)
            sys.stdout.flush()
        print


    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return model


def finetune(finetune_params):
    
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
    
    ########################
    # FINETUNING THE MODEL #
    ########################
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= finetune_batch_size
    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = model.build_finetune_functions(
                dataset=dataset, batch_size=finetune_batch_size,
                learning_rate=finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 50 * n_train_batches  # look as this many examples regardless
    patience_increase = 20.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.999  # a relative improvement of this much is
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
                msg = ('epoch %i, minibatch %i/%i, validation error %f %%' %
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
                    print
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    # print model.get_prediction([test_set_x.eval()[0]])

    print test_score
    # pdb.set_trace()
    return model, best_validation_loss, test_score, best_epoch




if __name__ == '__main__':
    pass