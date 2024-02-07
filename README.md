# Modeling_democracy
A Machine learning view point for Gender Affirmative Care(GAC) bills in the US based on Emotions and Belief's.

# EBA

        Emotional-Belief Expresion Coding Analysis
        Emotional-Belief Expression Analysis (EBA) is a coding method used in Advocacy Coalition Framework studies to better understand the emotions and beliefs of coalitions.
        It is a variation of past ACF belief coding and utilizes Philip Liefeld's Discourse Network Analyzer.

# Developemnt Pipeline
       https://github.com/marnenik/Modeling_democracy/assets/146151437/05e153a0-9240-4598-8f83-b6e41478d06c




# Initial software requirements for model Development.
- !pip install ktrain
- import pandas as pd
- import numpy as np
- import ktrain
- from ktrain import text
- from sklearn.model_selection import train_test_split
- !pip install torch
- !pip install transformers
- !pip install pandas
- !pip install datasets
- import torch
- from torch.utils.data import DataLoader, TensorDataset
- from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Fine Tuning the model for personal requirements
        Based on the data and analysis requirements we fine tune the model to make predictions, in our case predict emotions and beliefs of narrators.


# Different Fine tuned Models available for GAC Bills analysis.
# 1. TextBased emotion analysis model
        - It analyses the GAC bill based on the 8 implicit emotions available in the dataset.
        - The model is trained on the 8 different emotions and predicts future statements as one of the following.

# 2. Emotion Dyads Model
        - In the model is trained to consider both the emotion and the belief's of the person and predict future bills based on these constraints.
        - The model has 8 implicit emotions and multiple beliefs that dataset is divided based on.

# 3. Emotion and Belief Model
        - In this the model performs similarly in emotion prediction for the beliefs it is mainly divided into two main categories as primary beliefs and deep core beliefs.


# DATASET 
        Contains data made up of actors and their respective beliefs and emotions including the encoding of emotions. 

# BERT Model used for text based emotion analysis 

https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270


