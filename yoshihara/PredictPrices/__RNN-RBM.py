# coding: utf-8

# general library imports
import pudb, sys
from random import randint
import numpy as np
import pudb

# import theano
import theano
from theano import tensor as T
from theano.tensor import nnet
from theano.tensor.shared_randomstreams import RandomStreams

# import pylearn2
sys.path.extend(['/home/fujikawa/lib/python/other/pylearn2/pylearn2', '/home/fujikawa/lib/python/other/DeepLearningTutorials/code/', '/home/fujikawa/StockPredict/src/deeplearning/dataset'])
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



# import DL tutorial
# from rnnrbm import *

# import my library
from XOR import XOR
def build_rbm(v, W, bv, bh, k):
    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
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
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates

def shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent):
    W = shared_normal(n_visible, n_hidden, 0.01)
    bv = shared_zeros(n_visible)
    bh = shared_zeros(n_hidden)
    Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bu = shared_zeros(n_hidden_recurrent)

    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu  # learned parameters as shared
                                                # variables

    v = T.matrix()  # a training sequence
    u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden
                                         # units

    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence(v_t, u_tm1):
        bv_t = bv + T.dot(u_tm1, Wuv)
        bh_t = bh + T.dot(u_tm1, Wuh)
        generate = v_t is None
        if generate:
            v_t, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t,
                                           bh_t, k=25)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.
    (u_t, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1),
        sequences=v, outputs_info=[u0, None, None], non_sequences=params)
    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:],
                                                     k=15)
    updates_train.update(updates_rbm)

    # symbolic loop for sequence generation
    (v_t, u_t), updates_generate = theano.scan(
        lambda u_tm1, *_: recurrence(None, u_tm1),
        outputs_info=[None, u0], non_sequences=params, n_steps=200)

    return (v, v_sample, cost, monitor, params, updates_train, v_t,
            updates_generate)

class RnnRbm(RBM):

    def __init__(self, nvis = None, nhid = None,
            nhid_recurrent = None,
            vis_space = None,
            hid_space = None,
            transformer = None,
            irange=0.5, rng=None, init_bias_vis = None,
            init_bias_vis_marginals = None, init_bias_hid=0.0,
            base_lr = 1e-3, anneal_start = None, nchains = 100, sml_gibbs_steps = 1,
            random_patches_src = None,
            monitor_reconstruction = False):
        super(RnnRbm, self).__init__(nvis = nvis, nhid = nhid)

        self.lr = base_lr
        self.v_tm1 = sharedX(np.zeros(nhid_recurrent), name='v_tm1', borrow=True)
        self.v_t = sharedX(np.zeros(nhid_recurrent), name='v_tm1', borrow=True)
        # self.u_tm1 = sharedX(np.random.random(nhid_recurrent), name='u_tm1', borrow=True)
        # self.W = MatrixMul(sharedX(np.zeros([nvis, nhid]), name='W', borrow=True))
        self.W = sharedX(np.zeros([nvis, nhid]), name='W', borrow=True)
        self.Wuh = sharedX(np.random.random([nhid_recurrent, nhid]), name='Wuh', borrow=True)
        self.Wuv = sharedX(np.random.random([nhid_recurrent, nvis]), name='Wuv', borrow=True)
        self.Wvu = sharedX(np.random.random([nvis, nhid_recurrent]), name='Wvu', borrow=True)
        self.Wuu = sharedX(np.random.random([nhid_recurrent, nhid_recurrent]), name='Wuu', borrow=True)

        self.bv = sharedX(np.zeros(nvis), name='b_vt', borrow=True)
        self.bh = sharedX(np.zeros(nhid), name='b_ht', borrow=True)
        self.bu = sharedX(np.random.random(nhid_recurrent), name='b_ut', borrow=True)

        # self._params = [self.u_tm1, self.W.get_params()[0], self.Wuh, self.Wuv, self.Wvu, self.Wuu, self.bu, self.bv, self.bh]
        self._params = [self.W, self.Wuh, self.Wuv, self.Wvu, self.Wuu, self.bu, self.bv, self.bh]
        
        self.rng = np.random.RandomState(1001)
        self.rng = RandomStreams(seed=np.random.randint(1 << 30))
        v = T.matrix()  # a training sequence
        u0 = T.zeros((nhid_recurrent,))  

        (u_t, bv_t, bh_t), updates_train = theano.scan(
            lambda v_t, u_tm1, *_: self.recurrence(v_t, u_tm1),
            sequences=v, outputs_info=[u0, None, None], non_sequences=self._params)
        # v_mean, v_sample, = self.gibbs_step_for_v(v, self.rng)
        v_sample, cost, monitor, updates_rbm = self.build_rbm(v, k=nchains)
        
        updates_train.update(updates_rbm)
        print '----'
        # symbolic loop for sequence generation
        (v_t, u_t), updates_generate = theano.scan(
            lambda u_tm1, *_: self.recurrence(None, u_tm1),
            outputs_info=[None, u0], non_sequences=self._params, n_steps=2)
        print '----'
        gradient = T.grad(cost, self._params, consider_constant=[v_sample])

        updates_train.update(((p, p - lr * g) for p, g in zip(self._params, gradient)))
        self.train_function = theano.function([v], monitor, updates=updates_train)
        self.generate_function = theano.function([], v_t, updates=updates_generate)
    
    def build_rbm(self, v, k):
        print '--build_rbm'
        # v = T.matrix()
        print v
        chain, updates = theano.scan(lambda v: self.gibbs_step_for_v(v, self.rng)[1], outputs_info=[v], n_steps=k)
        print '---build_rbm'
        v_sample = chain[-1]
        mean_v = self.gibbs_step_for_v(v_sample, self.rng)[0]

        cost = self.ml_gradients(v, v_sample)
        p_ll = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
        p_ll = p_ll.sum() / v.shape[0]

        return v_sample, cost, p_ll, updates

    def gibbs_step_for_v(self, v, rng):
        print '-- gibbs_step_for_v'
        h_mean = self.mean_h_given_v(v)
        
        h_sample = rng.binomial(size = h_mean.shape, n = 1 , p = h_mean, dtype=theano.config.floatX)
        v_mean = self.mean_v_given_h(h_sample)
        v_sample = rng.binomial(size=v_mean.shape, n=1, p=v_mean, dtype=theano.config.floatX)
        # v_sample = self.sample_visibles([v_mean], v_mean.shape, rng)
        print '--- gibbs_step_for_v'
        print v_mean.eval()
        print v_sample.eval()
        return v_mean, v_sample

    def recurrence(self, v_t, u_tm1):
        bv_t = self.bv + T.dot(u_tm1, self.Wuv)
        bh_t = self.bh + T.dot(u_tm1, self.Wuh)
        generate = v_t is None
        if generate:
            print '-- generate'
            v_t, _, _, updates = self.build_rbm(T.zeros((self.nvis,)), k=25)
            print '--- generate'
        u_t = T.tanh(self.bu + T.dot(v_t, self.Wvu) + T.dot(u_tm1, self.Wuu))
        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

    def train(self, datasets, batch_size=2, num_epochs=1, lr=0.05):
        for epoch in xrange(num_epochs):
            costs = []
            for i in xrange(0, len(datasets), batch_size):
                cost = self.train_function(datasets[i: i + batch_size])
                costs.append(cost)
            print np.mean(costs)


    def _train(self, datasets, batch_size=2, num_epochs=1, lr=0.05):
        
        for epoch in xrange(num_epochs):
            costs = []
            # for dataset in datasets:
            #     for data in dataset:
            #         print data
            #     print ''
            
            for data in datasets[0]:

                sampler_updates = self.sampler.updates()
                pos_v = sharedX(data)
                neg_v = self.sampler.particles
                self.update_t(pos_v)
                grads = self.ml_gradients(pos_v, neg_v)
                # print [grad.eval() for grad in grads]
                # updates = dict([(param, param - self.lr * grad) for (param, grad) in zip(self._params, grads)])

                updates = [param - self.lr * grad for (param, grad) in zip(self._params, grads)]
                print type(updates)
                print type(self._params)
                self._params = updates
                # print [update.eval() for update in updates]
                # for update in updates:
                #     print update
                # # safe_update(self._params, updates)
                # safe_update(updates, sampler_updates)
                
                # for i, param in enumerate(self._params):
                #     print (param - self.lr * grads[i]).eval()
                #     self._params[i] = param - self.lr * grads[i]
                # ups = self.optimizer.updates()
                # safe_update(ups, sampler_updates)
                self.update_t(data)
                print [param.get_value() for param in self._params]
                # print self.gibbs_step_for_v(sharedX(data), self.rng)




    def get_bv(self):
        # return self.b_vt + self.Wuv.lmul(self.u_tm1)
        return self.bv
        return self.b_vt + T.dot(self.u_tm1, self.Wuv)

    def get_bh(self):
        return self.bh
        return self.b_ht + T.dot(self.u_tm1, self.Wuh)
        # return self.b_ht + tensor.dot(self.u_tm1, self.Wuh.lmul_T(self.u_tm1))

    def update_t(self, v):
        self.v_tm1 = v
        self.u_tm1 = T.dot(self.v_tm1, self.Wvu) + T.dot(self.u_tm1, self.Wuu) + self.b_ut
        self.b_vt = self.b_vt + T.dot(self.u_tm1, self.Wuv)
        self.b_ht = self.b_ht + T.dot(self.u_tm1, self.Wuh)
         
    def input_to_h_from_v(self, v):
        ###########  ???
        # self.update_t(v)
        print '****'
        if isinstance(v, T.Variable):
            return self.get_bh() + self.W
        else:
            return [self.input_to_h_from_v(vis) for vis in v]

    def input_to_v_from_h(self, h):
        if isinstance(h, T.Variable):
            return self.get_bv() + self.W.T
        else:
            return [self.input_to_v_from_h(hid) for hid in h]

    def free_energy_given_v(self, v):
        sigmoid_arg = self.input_to_h_from_v(v)
        return (-T.dot(v, self.get_bv()) -
                 nnet.softplus(sigmoid_arg).sum())

    def free_energy_given_h(self, h):
        sigmoid_arg = self.input_to_v_from_h(h)
        return (-T.dot(h, self.get_bh()) -
                nnet.softplus(sigmoid_arg).sum())

    def get_weights(self, borrow=False):
        weights ,= self.W.get_params()
        return weights.get_value(borrow=borrow)

    def get_weights_topo(self):
        return self.W.get_weights_topo()

    def ml_gradients(self, pos_v, neg_v):
        ml_cost = (self.free_energy_given_v(pos_v).mean() - self.free_energy_given_v(neg_v).mean())
        # grads = T.grad(ml_cost, self.get_params(), consider_constant=[pos_v, neg_v])
        # return grads
        return ml_cost
"""
    def input_to_h_from_v(self, v):
        ###########  ???
        # self.update_t(v)
        print '****'
        if isinstance(v, T.Variable):
            return self.get_bh() + self.W.lmul(v)
        else:
            return [self.input_to_h_from_v(vis) for vis in v]

    def input_to_v_from_h(self, h):
        if isinstance(h, T.Variable):
            return self.get_bv() + self.W.lmul_T(h)
        else:
            return [self.input_to_v_from_h(hid) for hid in h]

    def free_energy_given_v(self, v):
        sigmoid_arg = self.input_to_h_from_v(v)
        return (-T.dot(v, self.get_bv()) -
                 nnet.softplus(sigmoid_arg).sum())

    def free_energy_given_h(self, h):
        sigmoid_arg = self.input_to_v_from_h(h)
        return (-T.dot(h, self.get_bh()) -
                nnet.softplus(sigmoid_arg).sum())

    def get_weights(self, borrow=False):
        weights ,= self.W.get_params()
        return weights.get_value(borrow=borrow)

    def get_weights_topo(self):
        return self.W.get_weights_topo()

    def ml_gradients(self, pos_v, neg_v):
        ml_cost = (self.free_energy_given_v(pos_v).mean() - self.free_energy_given_v(neg_v).mean())
        # grads = T.grad(ml_cost, self.get_params(), consider_constant=[pos_v, neg_v])
        # return grads
        return ml_cost
"""
"""
class RnnRbm(RnnRbm):
    def __init__(self, n_hidden=150, n_hidden_recurrent=100, lr=0.001, dt=0.3):
        v, v_sample, cost, monitor, params, updates_train, v_t,updates_generate = build_rnnrbm(2, n_hidden, n_hidden_recurrent)
        self.dt = dt
        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update(((p, p - lr * g) for p, g in zip(params, gradient)))
        self.train_function = theano.function([v], monitor, updates=updates_train)
        self.generate_function = theano.function([], v_t, updates=updates_generate)
    def train(self, files, batch_size=100, num_epochs=200):

"""
###############################
####  Setting for dataset  ####
###############################
dataset = XOR(size=30, type='seq')
train_data = dataset.get_batch_design(10)
print len(train_data)

########################
####  Pre training  ####
########################
### First layer ###

rbm = RnnRbm(nvis=2, nhid=4, nhid_recurrent=3)
# rbm = RnnRbm(n_hidden=4, n_hidden_recurrent=3)
print ''

# for param in rbm.get_params():
#     print '------    ' + str(param) + '    -----'
#     print param.get_value()

# rbm.train(train_data)
# algorithm=SGD(learning_rate=.05, batch_size=2, termination_criterion=EpochCounter(3), cost = SMD_(corruptor=GaussianCorruptor(stdev=0.4)), monitoring_batches=2, train_iteration_mode = 'sequential', monitoring_dataset=dataset)
# algorithm=SGD(learning_rate=.05, batch_size=2, termination_criterion=EpochCounter(3), cost = PCD(), monitoring_batches=2, train_iteration_mode = 'sequential', monitoring_dataset=dataset)
# algorithm.setup(dataset=dataset, model=rbm)
# algorithm.train(dataset)
# sampler = BlockGibbsSampler(rbm, np.array(), numpy.random.RandomState(1234))   


# for param in rbm.get_params():
#     print '------    ' + str(param) + '    -----'
#     print param.get_value()


# for i in range(10):
#     algorithm.train(dataset)
#     print ''
#     for param in rbm.get_params():
#         print '------    ' + str(param) + '    -----'
#         print param.get_value()



# theano.tensor.matrix([0,0])
# print rbm.train_batch(dataset, 1)
# print rbm.learn_mini_batch(np.array([0,0]))
