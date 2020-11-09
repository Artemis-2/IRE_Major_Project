import os
import sys
import time
import tensorflow as tf
import data
import beam_search
import json
import pyrouge
import logging
import numpy as np
import util

PARAM_FLAGS = tf.app.flags.FLAGS



class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batcher, vocab):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    
    self._model.buildGraph()

    # Parameters passed from run_summarization
    self._batcher = batcher
    self._vocab = vocab
    self._model = model

    # Defined from tensorflow
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess)

    if PARAM_FLAGS.single_pass = True:
      ckpt_split = ckpt_path.split('-')
      self._decode_dir = os.path.join(PARAM_FLAGS.log_root, get_decode_dir_name( "ckpt-" + ckpt_split[len(ckpt_split)-1] ))

      if os.path.isdir(self._decode_dir):
        sys.exit("The single pass decode directory already exists ??")
        
    os.mkdir(os.path.join(PARAM_FLAGS.log_root, "decode"))
    self._decode_dir = os.path.join(PARAM_FLAGS.log_root, "decode")

    if PARAM_FLAGS.single_pass = True:
      # Make the dirs to contain output written in the correct format for pyrouge
      self._rouge_ref_dir = os.path.join(os.path.join(PARAM_FLAGS.log_root, "decode"), "reference")
      os.mkdir(self._rouge_ref_dir)
      self._rouge_dec_dir = os.path.join(os.path.join(PARAM_FLAGS.log_root, "decode"), "decoded")
      os.mkdir(self._rouge_dec_dir)


  def decode(self):
    """Decode examples until data is exhausted (if PARAM_FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    start_time = time.time()
    counter = 0
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      if batch is None: # finished decoding dataset in single_pass mode
        assert PARAM_FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)
        return

      original_article = batch.original_articles[0]  # string
      original_abstract = batch.original_abstracts[0]  # string
      original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

      article_withunks = data.show_art_oovs(original_article, self._vocab) # string
      para = batch.art_oovs[0]
      if PARAM_FLAGS.pointer_gen == False:
        para = None
      abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, para) # string

      # Run beam search to get best Hypothesis
      best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

      # Extract the output ids from the hypothesis and convert back to words
      best_hyp_use = best_hyp.tokens[1:]
      output_ids = [int(itm) for itm in best_hyp_use]
      decoded_words = data.outputids2words(output_ids, self._vocab, para)


      # Remove the [STOP] token from decoded_words, if necessary
      try:
        fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
      except:
        tf.logging.info("Error getting index of first stop in decode.py")
        sys.exit("Error getting index of first stop in decode.py")


      
      decoded_output = ' '.join(decoded_words) # single string

      if PARAM_FLAGS.single_pass:
        self.write_for_rouge(original_abstract_sents, decoded_words, counter) # write ref summary and decoded summary to file, to eval with pyrouge later
        counter += 1 # this is how many examples we've decoded
      else:
        print_results(article_withunks, abstract_withunks, decoded_output) # log output to screen
        # self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens) # write info to .json file for visualization tool

        # Load new checkpoint after every 90 seconds
       
        if time.time()- start_time > 90:
          tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', time.time()-start_time)
          _ = util.load_ckpt(self._saver, self._sess)
          start_time = time.time()

  def write_for_rouge(self, reference_sents, decoded_words, ex_index):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # First, divide decoded output into sentences
    decoded_sents = []
    while len(decoded_words) > 0:
      try:
        fst_period_idx = decoded_words.index(".")
      except ValueError: # there is text remaining that doesn't end in "."
        fst_period_idx = len(decoded_words)
      sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
      decoded_words = decoded_words[fst_period_idx+1:] # everything else
      decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    
    for itm in decoded_sents:
      itm.replace("<", "&lt;")
      itm.replace(">", "&gt;")

    for itm in reference_sents:
      itm.replace("<", "&lt;")
      itm.replace(">", "&gt;")
    
    # Write to file
    ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
      for idx,sent in enumerate(reference_sents):
        f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
      for idx,sent in enumerate(decoded_sents):
        f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

    tf.logging.info("Wrote example %i to file" % ex_index)


  


def print_results(article, abstract, decoded_output):
  """Prints the article, the reference summmary and the decoded summary to screen"""
  print("---------------------------------------------------------------------------")
  tf.logging.info('ARTICLE:  %s', article)
  tf.logging.info('REFERENCE SUMMARY: %s', abstract)
  tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
  print("---------------------------------------------------------------------------")




def rouge_eval(ref_dir, dec_dir):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
  """Log ROUGE results to screen and write to file.

  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_" + x + "_" + y
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str +=  key + ":" +  "%.4f with confidence interval (%.4f, %.4f)\n" % (val, val_cb, val_ce)
  tf.logging.info(log_str) # log to screen
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  tf.logging.info("Writing final ROUGE results to %s...", results_file)
  with open(results_file, "w") as f:
    f.write(log_str)

def get_decode_dir_name(ckpt_name):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""
  flg = 0

  if "train" in PARAM_FLAGS.data_path: 
    dataset = "train"
    flg = 1
  if "test" in PARAM_FLAGS.data_path: 
    dataset = "test"
    flg = 2
  if "val" in PARAM_FLAGS.data_path: 
    dataset = "val"
    flg = 3
  if flg == 0:
    sys.exit("PARAM_FLAGS.data_path doesnt have train test or val")
    
  dirname = "decode_" + dataset +"_" +str(PARAM_FLAGS.max_enc_steps) + "maxenc_" + str(PARAM_FLAGS.beam_size) + "beam_" + str( PARAM_FLAGS.min_dec_steps) + "mindec_" + str(PARAM_FLAGS.max_dec_steps) + "maxdec" 
  if ckpt_name is None:
    return dirname
  else:
    dirname += ckpt_name
  return dirname
