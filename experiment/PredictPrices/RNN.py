# coding: utf-8

import sys, pdb
sys.path.append('/home/fujikawa/lib/python/other/theano-rnn')
sys.path.append('/home/fujikawa/lib/python/other/theano-hf')

from rnn import MetaRNN
from rnn_minibatch import MetaRNN as MetaRNN_minibatch
from hf import SequenceDataset, hf_optimizer
import numpy as np

def train_RNN_minibatch(dataset=None, n_hidden=5000, steps=30, batch_size=5, n_epochs=100, optimizer='bfgs'):
    
    train_set_x, train_set_y = dataset.get_batch_design(0, steps, dataset.phase2['train'], type='numpy_dense', isInt=True)
    valid_set_x, valid_set_y = dataset.get_batch_design(0, steps, dataset.phase2['valid'], type='numpy_dense', isInt=True)
    test_set_x, test_set_y = dataset.phase2['test']['x'], dataset.phase2['test']['y']
    # train_set_y = np.asarray(valid_set_x, dtype=np.int32)
    # valid_set_x, valid_set_y = dataset.phase2['valid']['x'], dataset.phase2['valid']['y']
    # test_set_x, test_set_y = dataset.get_batch_design(0, batch_size, dataset.phase2['test'], type='numpy_dense')
    pdb.set_trace()
    print dataset.phase2_input_size
    model = MetaRNN_minibatch(n_in=dataset.phase2_input_size, n_hidden=n_hidden, n_out=1,
                    learning_rate=0.0001, learning_rate_decay=0.99,
                    n_epochs=n_epochs, activation='tanh',
                    batch_size=batch_size, output_type='binary'
                    )
    model.fit(train_set_x, train_set_y, validate_every=100, compute_zero_one=True, optimizer=optimizer)
    pdb.set_trace()
    return model

def test_binary(multiple_out=False, n_epochs=1000, optimizer='bfgs'):
    """ Test RNN with binary outputs. """
    n_hidden = 10
    n_in = 5
    if multiple_out:
        n_out = 2
    else:
        n_out = 1
    n_steps = 10
    n_seq = 10  # per batch
    n_batches = 50

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_steps, n_seq * n_batches, n_in)
    targets = np.zeros((n_steps, n_seq * n_batches, n_out))

    # whether lag 1 (dim 3) is greater than lag 2 (dim 0)
    targets[2:, :, 0] = np.cast[np.int](seq[1:-1, :, 3] > seq[:-2, :, 0])
    # pdb.set_trace()
    if multiple_out:
        # whether product of lag 1 (dim 4) and lag 1 (dim 2)
        # is less than lag 2 (dim 0)
        targets[2:, :, 1] = np.cast[np.int](
            (seq[1:-1, :, 4] * seq[1:-1, :, 2]) > seq[:-2, :, 0])

    model = MetaRNN_minibatch(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=0.005, learning_rate_decay=0.999,
                    n_epochs=n_epochs, batch_size=n_seq, activation='tanh',
                    output_type='binary')

    model.fit(seq, targets, validate_every=100, compute_zero_one=True,
              optimizer=optimizer)

    seqs = xrange(10)

    


def train_RNN(dataset=None, n_hidden=5000, batch_size=100):
    
    train_set_x, train_set_y = dataset.get_batch_design(0, batch_size, dataset.phase2['train'], type='numpy_dense')
    valid_set_x, valid_set_y = dataset.get_batch_design(0, batch_size, dataset.phase2['valid'], type='numpy_dense')
    test_set_x, test_set_y = dataset.phase2['test']['x'], dataset.phase2['test']['y']
    
    # valid_set_x, valid_set_y = dataset.phase2['valid']['x'], dataset.phase2['valid']['y']
    # test_set_x, test_set_y = dataset.get_batch_design(0, batch_size, dataset.phase2['test'], type='numpy_dense')

    print dataset.phase2_input_size
    model = MetaRNN(n_in=dataset.phase2_input_size, n_hidden=n_hidden, n_out=1,
                    learning_rate=0.00005, learning_rate_decay=0.999,
                    n_epochs=400, activation='tanh')
    model.fit(train_set_x, train_set_y, X_test=valid_set_x, Y_test=valid_set_y, validation_frequency=1)
    pdb.set_trace()
    return model

def train_RNN_hf(dataset=None, n_hidden=5000, batch_size=30, n_updates=100):

    train_set_x, train_set_y = dataset.get_batch_design(0, batch_size, dataset.phase2['train'], type='numpy_dense')
    # train_set_x, train_set_y = dataset.phase2['train']['x'], dataset.phase2['train']['y']
    valid_set_x, valid_set_y = dataset.phase2['valid']['x'], dataset.phase2['valid']['y']
    test_set_x, test_set_y = dataset.phase2['test']['x'], dataset.phase2['test']['y']
    train_set_x = [i for i in train_set_x]
    train_set_y = [i for i in train_set_y]
    gradient_dataset = SequenceDataset([train_set_x, train_set_y], batch_size=None,
                                       number_batches=100)
    cg_dataset = SequenceDataset([train_set_x, train_set_y], batch_size=None,
                                 number_batches=20)

    model = MetaRNN(n_in=dataset.phase2_input_size, n_hidden=n_hidden, n_out=1,
                    activation='tanh')

    opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                       s=model.rnn.y_pred,
                       costs=[model.rnn.loss(model.y)], h=model.rnn.h)

    opt.train(gradient_dataset, cg_dataset, num_updates=n_updates)

    pdb.set_trace()
    return model


if __name__ == '__main__':
    test_binary()
