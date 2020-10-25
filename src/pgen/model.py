import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import time
import os


class SummarizationModel(object):
    """
    A class that represents seq2seq model with additional 
    modifications such as pointer and generator.
    """
    def __init__(self,hps,vocab):

        self._vocab = vocab
        self._hps = hps
    def _add_placeholders(self):
        """
            Add placeholders to the graph.
        """
        hps = self._hps

        # encoder 
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
        if FLAGS.pointer_gen:
        self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
        self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        # decoder 
        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

        if hps.mode=="decode" and hps.coverage:
        self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')