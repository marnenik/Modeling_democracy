# Modeling_democracy
A Machine learning view point for Gender Affirmative Care(GAC) bills in the US based on Emotions and Belief's.

# Initial software requirements for model Development.
!pip install ktrain
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split

Different Models available for GAC Bills analysis.

1. TextBased emotion analysis model
        - It analyses the GAC bill based on the 8 implicit emotions available in the dataset.
        - The model is trained on the 8 different emotions and predicts future statements as one of the following.

2. Emotion Dyads Model
        - In the model is trained to consider both the emotion and the belief's of the person and predict future bills based on these constraints.
        - The model has 8 implicit emotions and multiple beliefs that dataset is divided based on.

3. Emotion and Belief Model
        - In this the model performs similarly in emotion prediction for the beliefs it is mainly divided into two main categories as primary beliefs and deep core beliefs.
