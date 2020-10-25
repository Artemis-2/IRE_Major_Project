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
    