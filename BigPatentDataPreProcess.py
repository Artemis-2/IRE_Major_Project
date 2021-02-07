
import json
import os
import re
import gzip
import sys
import argparse
import tensorflow_datasets.public_api as tfds
import stanza
import nltk
from nltk import word_tokenize
nltk.download('punkt')
import tensorflow as tf
from tensorflow.core.example import example_pb2

#Region -----Global Constants
_URL = "bigPatentData"

_DOCUMENT = "description"
_SUMMARY = "abstract"

_CPC_DESCRIPTION = {
    "a": "Human Necessities",
    "b": "Performing Operations; Transporting",
    "c": "Chemistry; Metallurgy",
    "d": "Textiles; Paper",
    "e": "Fixed Constructions",
    "f": "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
    "g": "Physics",
    "h": "Electricity",
    "y": "General tagging of new or cross-sectional technology"
}

_FIG_EXP1 = re.compile(r"(FIG.)\s+(\d)(,*)\s*(\d*)")
_FIG_EXP2 = re.compile(r"(FIGS.)\s+(\d)(,*)\s*(\d*)")
_FIG_EXP3 = re.compile(r"(FIGURE)\s+(\d)(,*)\s*(\d*)")

_LINE_NUM_EXP = re.compile(r"\[(\d+)\]")
_NON_EMPTY_LINES = re.compile(r"^\s*\[(\d+)\]")
_TABLE_HEADER = re.compile(r"^(\s*)TABLE\s+\d+(\s+(.*))?$")

_ENGLISH_WORDS = None

# End Region -----Global Constants

# Region Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read Input')
    parser.add_argument('--data_path', type=str, help='path to Unzipped Big Patent Dataset folder')
    parser.add_argument('--split_type', type=str, help='can be train, test, val')
  
    args = parser.parse_args()
    split_type = args.split_type
    input_path = args.input_path

    data = readData(input_path, split_type ,CPC_CODES)
	cleanData = getCleanPatentData(data)
	filename = "TokenizedData_"+split_type+".txt"
	with open(filename, 'w') as file:
    	file.write(json.dumps(cleanData))
	if(split_type="train"):
		makevocab(cleanData)

# End Region Main

# Region Preprocessing Functions
 def _get_english_words():
 	"""Load dictionary of common english words from NLTK."""
  	global _ENGLISH_WORDS
  	if not _ENGLISH_WORDS:
	    nltk = tfds.core.lazy_imports.nltk
	    resource_path = tfds.core.utils.resource_path(nltk)
	    data_path = os.fspath(resource_path / "nltk_data/corpora/words/en")
	    word_list = nltk.data.load(data_path, format="raw").decode("utf-8")
	    _ENGLISH_WORDS = frozenset(word_list.split("\n"))
  	return _ENGLISH_WORDS

def _remove_excessive_whitespace(text):
	return " ".join([w for w in text.split(" ") if w])


def _bigpatent_clean_abstract(text):
	"""Cleans the abstract text."""
    text = re.sub(r"[\(\{\[].*?[\}\)\]]", "", text).strip()
    text = _remove_excessive_whitespace(text)
    return text


def _bigpatent_remove_referenecs(text):
	"""Remove references from description text."""
  	text = _FIG_EXP1.sub(r"FIG\2 ", text)
  	text = _FIG_EXP2.sub(r"FIG\2 ", text)
  	text = _FIG_EXP3.sub(r"FIG\2 ", text)
  	return text

def _bigpatent_get_list_of_non_empty_lines(text):
	"""Remove non-empty lines."""
  	# Split into lines
  	# Remove empty lines
  	# Remove line numbers
  	return [
  		_NON_EMPTY_LINES.sub("", s).strip()
      	for s in text.strip().splitlines(True)
      	if s.strip()
  	]


def _bigpatent_remove_tables(sentences):
	"""Remove Tables from description text."""
  	# Remove tables from text
  	new_sentences = []
  	i = 0
  	table_start = 0
  	# A table header will be a line starting with "TABLE" after zero or more
  	# whitespaces, followed by an integer.
  	# After the integer, the line ends, or is followed by whitespace and
  	# description.
  	while i < len(sentences):
    	sentence = sentences[i]
    	if table_start == 0:
    		# Not inside a table
      		# Check if it's start of a table
      		if _TABLE_HEADER.match(sentence):
        		table_start = 1
      		else:
        	new_sentences.append(sentence)
    	elif table_start == 1:
      		words = sentence.strip("\t").split(" ")
      		num_eng = 0
		    for w in words:
		    	if not w.isalpha():
		    		continue
		        if w in _get_english_words():
		        	num_eng += 1
		        	if num_eng > 20:
			            # Table end condition
			            table_start = 0
			            new_sentences.append(sentence)
		            break
    	i += 1
	return new_sentences


def _bigpatent_remove_lines_with_less_words(sentences):
	"""Remove sentences with less than 10 words."""
	new_sentences = []
	for sentence in sentences:
    	words = set(sentence.split(" "))
    	if len(words) > 10:
      		new_sentences.append(sentence)
  	return new_sentences


def _bigpatent_clean_description(text):
	"""Clean the description text."""
  	# split the text by newlines, keep only non-empty lines
  	sentences = _bigpatent_get_list_of_non_empty_lines(text)
  	# remove tables from the description text
  	sentences = _bigpatent_remove_tables(sentences)
  	# remove sentences with less than 10 words
 	# sentences = _bigpatent_remove_lines_with_less_words(sentences)
  	#text = "\n".join(sentences)
  	# remove references like FIG. 8, FIGS. 8, 8, FIG. 8-d
  	text = _bigpatent_remove_referenecs(text)
  	# remove excessive whitespace
  	text = _remove_excessive_whitespace(text)
  	return text
# End Region Preprocessing Functions

# Region

def preprocessDescription(descText):
	return _bigpatent_clean_description(descText)

def preprocessAbstract(absText):
    return _bigpatent_clean_abstract(absText)

def readData(input_path,split_type,cpc_codes):
    #file_names = os.listdir(os.path.join(input_path,split_type,cpc_code))
    # reading one of the gz files.
    #file_name = file_names[0]
    data=[]
    for code in cpc_codes:
        file_list = os.listdir(os.path.join(input_path,split_type,code))
        
        for file_name in file_list:
            with gzip.open(os.path.join(input_path,split_type,code,file_name),'r') as fin:
                for row in fin:
                	json_obj = json.loads(row)
                    data.append(json_obj)
                    
    return data

def getCleanPatentData(data):
    
    for patentData in data:
        cleanData = []
        patentData["abstract"]= word_tokenize(preprocessAbstract(patentData["abstract"]))
        patentData["description"] = word_tokenize(preprocessDescription(patentData["description"]))
        patentData["application_number"] = patentData["application_number"]
        patentData["publication_number"] = patentData["publication_number"]
        cleanData.append(patentData)
        return cleanData

def makevocab(data):
    vocab={}
    for patent in data:
        abstract = patent["abstract"]
        desc = patent["description"]
        for word in abstract:
            if word in vocab:
                vocab[word]+=1
            else:
                vocab[word]=1
        for word in desc:
            if word not in vocab.keys():
                vocab[word]=1
            else:
                vocab[word]+=1
        
    with open('vocab.txt', 'w') as file:
        file.write(json.dumps(vocab))
# End Region

