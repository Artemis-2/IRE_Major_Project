import tensorflow as tf
import numpy as np
import data

FLAGS = tf.app.flags.FLAGS

class Hypothesis(object):
  def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):

    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.attn_dists = attn_dists
    self.p_gens = p_gens
    self.coverage = coverage

  def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      attn_dists = self.attn_dists + [attn_dist],
                      p_gens = self.p_gens + [p_gen],
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    
    return sum(self.log_probs)


  @property
  def avg_log_prob(self):
    
    return self.log_prob / len(self.tokens)

