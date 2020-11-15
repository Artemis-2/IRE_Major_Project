# imports
import os
import time
import numpy as np
import tensorflow as tf
from attention import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

# Global Variables
FLAGS = tf.app.flags.FLAGS
TRAIN_DIR = os.path.join(FLAGS.out_dir, "TRAIN")

class SummarizationModel(object):
    """
    class for seq2seq, pointer-generator and coverage architectures
    """
    # initialized by main.py -- starting point of initialization of training process.
    def __init__(self, hyperParams, vocabObj):

        self._hyperParams = hyperParams

        self._vocabObj = vocabObj

    def _addPlaceHoldersToTFGraph(self):

        hyperParams = self._hyperParams

        # setting up necessary placeholders for encoder
        if FLAGS.pointer_gen == True:

            self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')
            self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hyperParams.batch_size, None], name='enc_batch_extend_vocab')
        
        self._enc_batch = tf.placeholder(tf.int32, [hyperParams.batch_size, None], name='enc_batch')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hyperParams.batch_size, None], name='enc_padding_mask')
        self._enc_lens = tf.placeholder(tf.int32, [hyperParams.batch_size], name='enc_lens')

        # setting up necessary placeholders for decoder
        if hyperParams.mode=="decode" and hyperParams.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [hyperParams.batch_size, None], name='prev_coverage')
        self._dec_batch = tf.placeholder(tf.int32, [hyperParams.batch_size, hyperParams.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hyperParams.batch_size, hyperParams.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hyperParams.batch_size, hyperParams.max_dec_steps], name='dec_padding_mask')
    
    
    def _getFeedDictionary(self,batch,onlyEncoding=False): #_make_feed_dict

        """
        IN:
            batch: Batch object
            onlyEncoding: Boolean. If True, only feed the parts needed for the encoder.
        OUT:
            Returns a feed dictionary mapping parts of the batch to the appropriate placeholders.
        """

        feedDictionary = dict()
        if FLAGS.pointer_gen == True:
            feedDictionary[self._max_art_oovs] = batch.max_art_oovs
            feedDictionary[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
        if onlyEncoding == False:
            feedDictionary[self._dec_batch] = batch.dec_batch
            feedDictionary[self._dec_padding_mask] = batch.dec_padding_mask
            feedDictionary[self._target_batch] = batch.target_batch
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        

        return feedDictionary


    def _loadEncoder(self, encoder_inputs, seq_len): #_add_encoder()
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
        encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
        seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
        encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. 2*hidden_dim since it concatenates the backward and forward states
        forward_state, backward_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('encoder'):
            lstm_forward_cell = tf.contrib.rnn.LSTMCell(self._hyperParams.hidden_dim, state_is_tuple=True , initializer=self.rand_unif_init)
            lstm_backward_cell = tf.contrib.rnn.LSTMCell(self._hyperParams.hidden_dim,state_is_tuple=True,initializer=self.rand_unif_init)
            (encoder_outputs, (forward_state, backward_state)) = tf.nn.bidirectional_dynamic_rnn(lstm_forward_cell, lstm_backward_cell, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    
        return encoder_outputs, forward_state, backward_state

    
    def _loadDecoder(self, inputs): # _add_decoder
        """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

        Args:
        inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

        Returns:
        outputs: List of tensors; the outputs of the decoder
        out_state: The final state of the decoder
        attn_dists: A list of tensors; the attention distributions
        p_gens: A list of scalar tensors; the generation probabilities
        coverage: A tensor, the current coverage vector
        """
    
        if self._hyperParams.mode=="decode" and self._hyperParams.coverage == True:
            prev_coverage = self.prev_coverage 
        
        else:
            prev_coverage = None  ## In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
        
        LSTMcell = tf.contrib.rnn.LSTMCell(self._hyperParams.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
        

        outputs, out_state, attentionDistributions, genProb, coverage = attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell, initial_state_attention=(self._hyperParams.mode=="decode"), pointer_gen=self._hyperParams.pointer_gen, use_coverage=self._hyperParams.coverage, prev_coverage=prev_coverage)

        return outputs, out_state, attentionDistributions, genProb, coverage

    def _reduceStates(self, forward_states, backward_states): #_reduce_states
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

        Args:
        forward_states: LSTMStateTuple with hidden_dim units.
        backward_states: LSTMStateTuple with hidden_dim units.

        Returns:
        state: LSTMStateTuple with hidden_dim units.
        """
        
        hidden_dim = self._hyperParams.hidden_dim

        with tf.variable_scope('reduce_final_st'):

            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[forward_states.c, backward_states.c]) # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[forward_states.h, backward_states.h]) # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
        return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state
    
    def _loadEmb2TB(self, embedding_var): #_add_emb_vis
    
        vocab_metadata_path = os.path.join(TRAIN_DIR, "vocab_metadata.tsv")
        self._vocab.write_metadata(vocab_metadata_path) # write metadata file
        summWriter = tf.summary.FileWriter(TRAIN_DIR)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summWriter, config)

    def _loadSeq2Seq(self): #_add_seq2seq
        """loads the whole seq2seq model to the tf computational graph."""
        
        vocab_size = self._vocabObj.size() # size of the vocabulary
        hyperParams = self._hyperParams
            
        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hyperParams.trunc_norm_init_std)
            self.rand_unif_init = tf.random_uniform_initializer(-hyperParams.rand_unif_init_mag, hyperParams.rand_unif_init_mag, seed=123)

        # Add embedding matrix (shared by the encoder and decoder inputs)
        with tf.variable_scope('embedding'):

            embedding = tf.get_variable('embedding', [vocab_size, hyperParams.emb_dim], initializer=self.trunc_norm_init,dtype=tf.float32)            
            emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
            emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)
            if hyperParams.mode=="train": 
                self._loadEmb2TB(embedding) # add embedding to tensorboard
        
        # Add the encoder.
        encoderOutputs, forward_state, backward_state = self._loadEncoder(emb_enc_inputs, self._enc_lens)
        self._enc_states = encoderOutputs

        # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
        self._dec_in_state = self._reduceStates(forward_state, backward_state)

        # Add the decoder.
        with tf.variable_scope('decoder'):
            decoderOutputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._loadDecoder(emb_dec_inputs)

        # Add the output projection to obtain the vocabulary distribution
        with tf.variable_scope('output_projection'):
            w = tf.get_variable('w', [hyperParams.hidden_dim, vocab_size], dtype=tf.float32, initializer=self.trunc_norm_init)
            w_t = tf.transpose(w)
            v = tf.get_variable('v', [vocab_size], dtype=tf.float32, initializer=self.trunc_norm_init)
            vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
            for i,output in enumerate(decoderOutputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

            vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vocab_size) arrays. The words are in the order they appear in the vocabulary file.

        if FLAGS.pointer_gen == False:
            final_dists = vocab_dists   # i.e. P(w) = P_vocab(w)
        elif FLAGS.pointer_gen == True:
            final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)

        # loss calculation
        if hyperParams.mode == 'train' or hyperParams.mode == 'eval':
            # Calculate the loss
            with tf.variable_scope('loss'):
                if FLAGS.pointer_gen == False:
                    # simple seq2seq model
                    self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # Note that this applies softmax internally and returns the output
                
                elif FLAGS.pointer_gen == True:
                    # Calculate the loss per step
                    
                    loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
                    batch_nums = tf.range(0, limit=hyperParams.batch_size) # shape (batch_size)
                    for dec_step, dist in enumerate(final_dists):
                        targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
                        indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
                        gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
                        losses = -tf.log(gold_probs)
                        loss_per_step.append(losses)

                    # Apply dec_padding_mask and get loss
                    self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

                

                tf.summary.scalar('loss', self._loss)

                # Calculate coverage loss from the attention distributions
                if hyperParams.coverage == True:
                    with tf.variable_scope('coverage_loss'):
                        self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                        tf.summary.scalar('coverage_loss', self._coverage_loss)
                    self._total_loss = self._loss + hyperParams.cov_loss_wt * self._coverage_loss
                    tf.summary.scalar('total_loss', self._total_loss)

        if hyperParams.mode == "decode":
            # We run decode beam search mode one decoder step at a time
            assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
            final_dists = final_dists[0]
            topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hyperParams.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
            self._topk_log_probs = tf.log(topk_probs)

    def _loadTrainOp(self): #_add_train_op
        
        """Sets self._train_op, the op to run for training."""
    
        if self._hyperParams.coverage == True:
            loss_to_minimize = self._total_loss # Will include the coverage loss -- remember loss is -log P(w) + lambda* min(a_i,c_i)
        else:
            loss_to_minimize = self._loss

        # calculate the gradients with respect to the loss variable for each of the learnable parameters

        tvars = tf.trainable_variables()

        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip Gradients
        with tf.device("/gpu:0"):

        grads, global_norm = tf.clip_by_global_norm(gradients, self._hyperParams.max_grad_norm)

        # Add a info to the tensorboard
        tf.summary.scalar('global_norm', global_norm)

        # Apply adagrad optimizer
        optimizer = tf.train.AdagradOptimizer(self._hyperParams.lr, initial_accumulator_value=self._hyperParams.adagrad_init_acc)
        with tf.device("/gpu:0"):
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


    def buildGraph(self):
        # pdb.set_trace()
        """
        Setup the whole architecture, optimizer etc onto the computational graph
        """
        tf.logging.info('Constructing the Computational Graph...')
        t0 = time.time()

        self._addPlaceHoldersToTFGraph()

        with tf.device("/gpu:0"):
        
            self._loadSeq2Seq()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self._hyperParams.mode == 'train':

            self._loadTrainOp()

        self._summaries = tf.summary.merge_all()

        t1 = time.time()

        tf.logging.info('Construction of Computation Graph took : %i seconds', t1 - t0)

    def runTrainStep(self, sess, batch): #run_train_step
        # pdb.set_trace()
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feedDictionary = self._getFeedDictionary(batch)
        results = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }

        if self._hyperParams.coverage == True:
            results['coverage_loss'] = self._coverage_loss
        return sess.run(results, feedDictionary)

    def runEvalStep(self, sess, batch): #run_eval_step
        # pdb.set_trace()
        """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
        feedDictionary = self._getFeedDictionary(batch)
        results = {
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self._hyperParams.coverage == True:
            results['coverage_loss'] = self._coverage_loss
        return sess.run(results, feedDictionary)

    def runEncoder(self, sess, batch): #run_encoder
        
        feed_dict = self._getFeedDictionary(batch, just_enc=True) # feed the batch into the placeholders
        (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        return enc_states, dec_in_state


    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):

        # pdb.set_trace()
        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # dims: [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # dims: [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
        }

        to_return = {
        "ids": self._topk_ids,
        "probs": self._topk_log_probs,
        "states": self._dec_out_state,
        "attn_dists": self.attn_dists
        }

        if FLAGS.pointer_gen == True:
            feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed[self._max_art_oovs] = batch.max_art_oovs
            to_return['p_gens'] = self.p_gens

        if self._hyperParams.coverage == True:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed) 

        
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in range(beam_size)]

        
        assert len(results['attn_dists'])==1
        attn_dists = results['attn_dists'][0].tolist()

        if FLAGS.pointer_gen:
            
            assert len(results['p_gens'])==1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]

        
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


    def _mask_and_avg(values, padding_mask):
        dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
        values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
        values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
        return tf.reduce_mean(values_per_ex) # overall average


    def _coverage_loss(attn_dists, padding_mask):
    
        coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
        covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
        for a in attn_dists:
            covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
            covlosses.append(covloss)
            coverage_loss += a # update the coverage vector
        coverage_loss = _mask_and_avg(covlosses, padding_mask)
        return coverage_loss

    def _calc_final_dist(self, vocab_dists, attn_dists): #_calc_final_dist
        
        pdb.set_trace()
        with tf.variable_scope('final_distribution'):
        # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
            vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
            attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]

            # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
            extended_vsize = self._vocab.size() + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
            extra_zeros = tf.zeros((self._hyperParams.batch_size, self._max_art_oovs))
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

            
            batch_nums = tf.range(0, limit=self._hyperParams.batch_size) # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
            batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
            indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
            shape = [self._hyperParams.batch_size, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

           
            final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

            return final_dists