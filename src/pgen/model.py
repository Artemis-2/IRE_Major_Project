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

    def _add_seq2seq(self):
    
        hps = self._hps
        vsize = self._vocab.size() 

        with tf.variable_scope('seq2seq'):

        self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)


        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            if hps.mode=="train": self._add_emb_vis(embedding) 
            emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch) 
            emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] 

        
        enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
        self._enc_states = enc_outputs

        
        self._dec_in_state = self._reduce_states(fw_st, bw_st)

        
        with tf.variable_scope('decoder'):
            decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(emb_dec_inputs)

        
        with tf.variable_scope('output_projection'):
            w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
            w_t = tf.transpose(w)
            v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
            vocab_scores = [] 
            for i,output in enumerate(decoder_outputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) 

            vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] 


        if FLAGS.pointer_gen:
            final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
        else: 
            final_dists = vocab_dists



        if hps.mode in ['train', 'eval']:
            # LOSS
            with tf.variable_scope('loss'):
            if FLAGS.pointer_gen:
            
                loss_per_step = [] 
                batch_nums = tf.range(0, limit=hps.batch_size) 
                for dec_step, dist in enumerate(final_dists):
                targets = self._target_batch[:,dec_step] 
                indices = tf.stack( (batch_nums, targets), axis=1) 
                gold_probs = tf.gather_nd(dist, indices) 
                losses = -tf.log(gold_probs)
                loss_per_step.append(losses)

                
                self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

            else: 
                self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask)

            tf.summary.scalar('loss', self._loss)

            
            if hps.coverage:
                with tf.variable_scope('coverage_loss'):
                self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                tf.summary.scalar('coverage_loss', self._coverage_loss)
                self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
                tf.summary.scalar('total_loss', self._total_loss)

        if hps.mode == "decode":
        
        assert len(final_dists)==1 
        final_dists = final_dists[0]
        topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size*2) 
        self._topk_log_probs = tf.log(topk_probs)
    