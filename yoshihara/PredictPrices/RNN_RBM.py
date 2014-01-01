# coding: utf-8

# general library imports
import pudb, sys
from random import randint
import numpy as np
import pdb

# import theano
import theano
from theano import tensor as T
from theano.tensor import nnet
from theano.tensor.shared_randomstreams import RandomStreams

# import pylearn2
"""
sys.path.extend(['/home/fujikawa/lib/python/other/pylearn2/pylearn2', '/home/fujikawa/StockPredict/src/deeplearning/dataset'])
from pylearn2.models import mlp
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.models.rbm import RBM, BlockGibbsSampler
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.costs.cost import Cost
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.corruption import GaussianCorruptor
from pylearn2.utils import as_floatX, safe_update, sharedX, safe_union
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.linear.matrixmul import MatrixMul

# import my library
from XOR import XOR
"""
#Don't use a python long as this don't work on 32 bits computers.
np.random.seed(0xbeef)
rng = RandomStreams(seed=np.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False



class RnnRbm(object):
#class RnnRbm(RBM):
    def __init__(self, input=None, n_visible=2, n_hidden=4, n_hidden_recurrent=3, lr=1.0):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_hidden_recurrent = n_hidden_recurrent
        
        self.W = theano.shared(np.zeros([n_visible, n_hidden]), name='W', borrow=True)
        self.Wuh = theano.shared(np.random.random([n_hidden_recurrent, n_hidden]), name='Wuh', borrow=True)
        self.Wuv = theano.shared(np.random.random([n_hidden_recurrent, n_visible]), name='Wuv', borrow=True)
        self.Wvu = theano.shared(np.random.random([n_visible, n_hidden_recurrent]), name='Wvu', borrow=True)
        self.Wuu = theano.shared(np.random.random([n_hidden_recurrent, n_hidden_recurrent]), name='Wuu', borrow=True)
        self.bv = theano.shared(np.zeros(n_visible), name='b_vt', borrow=True)
        self.bh = theano.shared(np.zeros(n_hidden), name='b_ht', borrow=True)
        self.bh_t = theano.shared(np.zeros(n_hidden), name='bh_t', borrow=True)
        self.bu = theano.shared(np.random.random(n_hidden_recurrent), name='b_ut', borrow=True)

        self.params = self.W, self.bv, self.bh, self.Wuh, self.Wuv, self.Wvu, self.Wuu, self.bu  # learned parameters as shared
                                                    # variables
        self.v = input
        if not input:
            self.v = T.matrix('v')
        u0 = T.zeros((self.n_hidden_recurrent,))  # initial value for the RNN hidden
                                             # units
        self.u_tm1 = u0
       
    def get_cost_updates(self,lr=1.0):
        (v, v_sample, cost, monitor, params, updates_train, v_t,
         updates_generate,bh_t,updates_bh_t) = self.build_rnnrbm(self.n_visible, self.n_hidden,
                                                self.n_hidden_recurrent)
        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update(((p, p - g * T.cast(lr,dtype=theano.config.floatX) ) for p, g in zip(params,
                                                                gradient)))
        """
        self.train_function = theano.function([v], monitor,
                                               updates=updates_train)
        self.bh_t_function = theano.function([v] ,bh_t,
                                                updates = updates_bh_t)
        self.generate_function = theano.function([], v_t,
                                                 updates=updates_generate)
        self.updates_g = updates_generate
        """
        self.bh_t = bh_t
        return monitor,updates_train,bh_t,updates_bh_t

    def build_rnnrbm(self, n_visible, n_hidden, n_hidden_recurrent):
        u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden
                                             # units

        def recurrence(v_t, u_tm1):
            bv_t = self.bv + T.dot(u_tm1, self.Wuv)
            bh_t = self.bh + T.dot(u_tm1, self.Wuh)
            generate = v_t is None
            if generate:
                v_t, _, _, updates = self.build_rbm(T.zeros((n_visible,)), self.W, bv_t,
                                               bh_t, k=25)
            u_t = T.tanh(self.bu + T.dot(v_t, self.Wvu) + T.dot(u_tm1, self.Wuu))
            return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

        (u_t, bv_t, bh_t), updates_train = theano.scan(
            lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1),
            sequences=self.v, outputs_info=[u0, None, None], non_sequences=self.params)
        v_sample, cost, monitor, updates_rbm = self.build_rbm(self.v, self.W, bv_t[:], bh_t[:],
                                                         k=15)
        
        updates_bh_t = updates_train.copy()

        updates_train.update(updates_rbm)

        # symbolic loop for sequence generation
        (v_t, u_t), updates_generate = theano.scan(
            lambda u_tm1, *_: recurrence(None, u_tm1),
            outputs_info=[None, u0], non_sequences=self.params, n_steps=1)

        return (self.v, v_sample, cost, monitor, self.params, updates_train, v_t,
                updates_generate,bh_t,updates_bh_t)


    def get_ut(self, v_t):
        bv_t = self.bv + T.dot(self.u_tm1, self.Wuv)
        bh_t = self.bh + T.dot(self.u_tm1, self.Wuh)
        u_t = T.tanh(self.bu + T.dot(v_t, self.Wvu) + T.dot(self.u_tm1, self.Wuu))
        self.u_tm1 = u_t
        return u_t

    def build_rbm(self, v, W, bv, bh, k):

        def gibbs_step(v):
            mean_h = T.nnet.sigmoid(T.dot(v, self.W) + bh)
            h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                             dtype=theano.config.floatX)
            mean_v = T.nnet.sigmoid(T.dot(h, self.W.T) + bv)
            v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                             dtype=theano.config.floatX)
            return mean_v, v

        chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                     n_steps=k)
        v_sample = chain[-1]

        mean_v = gibbs_step(v_sample)[0]
        monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
        monitor = monitor.sum() / v.shape[0]

        def free_energy(v):
            return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, self.W) + bh)).sum()
        cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

        return v_sample, cost, monitor, updates

    def train(self, datasets, batch_size=2, num_epochs=1, lr=0.5):
        for epoch in xrange(num_epochs):
            costs = []
            for data in datasets:
                cost = self.train_function(data)
                costs.append(cost)
            print np.mean(costs)
            # self.printParams()

    def printParams(self):
        for param in self.params:
            print '------    ' + str(param) + '    -----'
            print param.get_value()
    
    def generate(self, show=True):
        '''Generate a sample sequence, plot the resulting piano-roll and save
it as a MIDI file.

filename : string
  A MIDI file will be created at this location.
show : boolean
  If True, a piano-roll of the generated sequence will be shown.'''
        print self.generate_function()

        """
        piano_roll = self.generate_function()
        midiwrite(filename, piano_roll, self.r, self.dt)
        if show:
            extent = (0, self.dt * len(piano_roll)) + self.r
            pylab.figure()
            pylab.imshow(piano_roll.T, origin='lower', aspect='auto',
                         interpolation='nearest', cmap=pylab.cm.gray_r,
                         extent=extent)
            pylab.xlabel('time (s)')
            pylab.ylabel('MIDI note number')
            pylab.title('generated piano-roll')
        """

def test_rnnrbm(dataset, batch_size=2, num_epochs=10):
    model = RnnRbm()
    model.train(dataset, batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    
    dataset = XOR(size=30, type='seq')
    train_data = dataset.get_batch_design(10)
    # print train_data
    model = test_rnnrbm(train_data)
    # model.printParams()
    model.generate()
    pdb.set_trace()

