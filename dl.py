#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pyspark
import sparknlp
import pandas as pd
import string, re

from pymongo import MongoClient

from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline, Transformer

from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import ArrayType, StringType

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
eng_stopwords = stopwords.words('english')
punctuation = string.punctuation+'“’”...—…‼‘'

spark = sparknlp.start(m1=True)


# ## Fetch Data fro MongoDB

# In[2]:


client = MongoClient('mongodb://yongtai:taiyong@test.asknyu.com:27017/')
proj_db = client.project
constraint_train = proj_db.constraint_train
constraint_val = proj_db.constraint_val
constraint_test = proj_db.constraint_test


# In[3]:


train_pandasDF = pd.DataFrame(list(constraint_train.find())).drop(['_id'], axis=1)
trainData = spark.createDataFrame(train_pandasDF) 

val_pandasDF = pd.DataFrame(list(constraint_val.find())).drop(['_id'], axis=1)
valData = spark.createDataFrame(val_pandasDF) 

test_pandasDF = pd.DataFrame(list(constraint_test.find())).drop(['_id'], axis=1)
testData = spark.createDataFrame(test_pandasDF) 


# ## Preprocessing

# In[67]:


class CustomTransformer(Transformer):
    # lazy workaround - a transformer needs to have these attributes
    _defaultParamMap = dict()
    _paramMap = dict()
    _params = dict()

class NullDropper(CustomTransformer):
  def __init__(self, cols=None):
    self.cols = cols

  def _transform(self, data):
    return data.dropna()

class LabelEncoder(CustomTransformer):
  def __init__(self, cols=None):
    self.cols = cols

  def _transform(self, data):
    return data.withColumn("label", when(col("label")=="real", 0.0).otherwise(1.0))

class Cleaner(CustomTransformer):
  def __init__(self, cols=None):
    self.cols = cols

  def _transform(self, data):
    def filter_out_urls(words):
      # eliminate nulls and blanks
      newWords = []
      for word in words.split(" "):
          if not word.startswith("https:"):
              newWords.append(word)
      return " ".join(newWords)

    udf_filter_urls = udf(filter_out_urls, StringType())
    return data.withColumn("text", udf_filter_urls(col("tweet")))


# ### With Universal Sentence Encoder
# Best validation f-1 score: **91.82%**

# In[74]:


# actual content is inside description column
document = DocumentAssembler()                  .setInputCol("tweet")                  .setOutputCol("document")

# we can also use sentece detector here if we want to train on and get predictions for each sentence
use = UniversalSentenceEncoder.pretrained("tfhub_use_lg", "en")                   .setInputCols("document")                   .setOutputCol("sentence_embeddings")

# the classes/labels/categories are in category column
classsifierdl = ClassifierDLApproach()                  .setInputCols(["sentence_embeddings"])                  .setOutputCol("category")                  .setLabelColumn("label")                  .setMaxEpochs(1000)                  .setLr(0.001)                  .setBatchSize(32)                  .setEnableOutputLogs(True)                   .setOutputLogsPath('logs')


# In[75]:


trainPipeline = Pipeline(stages = [
    #nullDroper, 
    #labelEncoder, 
    document,
    use,
    classsifierdl
])

model = trainPipeline.fit(trainData)


# In[ ]:


model.save("model")


# #### Evaluation
