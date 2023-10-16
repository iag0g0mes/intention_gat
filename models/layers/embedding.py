
import os
import sys
sys.path.append(os.path.abspath('../../.'))

import numpy as np
import pandas as pd

import tensorflow as tf

from typing import Dict, NoReturn, List, Tuple, Union

from utils.weighted_categorical_crossentropy import weighted_categorical_crossentropy

"""
Embedding Layer

- responsible for embedding node states using two
    different metrics:

    1) per class

        1.1.) surrounding vehicle
            - LSTM

        1.2) target_vehicle
            - LSTM 
        
        1.3) pedestrian
            - LSTM
        
        1.4) cyclist
            - LSTM

    2) per feature:

        2.1) historical trajectory
            - LSTM
        2.2) lane deviation
            - LSTM
        2.3) lane geometry
            - LSTM
        2.4) road geometry
            - CNN
"""

class FeaturesEmbedding(tf.keras.layers.Layer):
    
    CLASS_SR_VEHICLE = 1
    CLASS_TARGET_VEHICLE = 2
    CLASS_PEDESTRIAN = 3
    CLASS_CYCLIST = 4

    FEATURE_HIST_TRAJ = 5
    FEATURE_LANE_DEV = 6
    FEATURE_LANE_GEO = 7
    FEATURE_ROAD_GEO = 8

    def __init__(
        self,
        embedding_type:str,
        dropout:float=0.0,
        return_sequences:bool=False,
        **kwargs
    ):
        
        super(FeaturesEmbedding, self).__init__(**kwargs)

        if embedding_type<=0 or embedding_type>8:
            raise ValueError(("embedding_type not found!"
                              " It must be a number in the range [1, 8]."))

        self.embedding_type=embedding_type
        self.return_sequences=return_sequences
        self.dropout = dropout

        self.emb = None
        self.drop_l = None

        if self.embedding_type == self.CLASS_SR_VEHICLE:
            self.emb = tf.keras.layers.LSTM(
                units = 64,
                dropout=0.2,
                activation="tanh",
                kernel_initializer='glorot_uniform',
                return_sequences=self.return_sequences,
                name=self.name
            )
        elif self.embedding_type == self.CLASS_TARGET_VEHICLE:
            self.emb = tf.keras.layers.LSTM(
                units = 64,
                dropout=0.2,
                activation="tanh",
                kernel_initializer='glorot_uniform',
                return_sequences=self.return_sequences,
                name=self.name
            )
        elif self.embedding_type == self.CLASS_PEDESTRIAN:
            self.emb = tf.keras.layers.LSTM(
                units = 64,
                dropout=0.2,
                activation="tanh",
                kernel_initializer='glorot_uniform',
                return_sequences=self.return_sequences,
                name=self.name
            )
        elif self.embedding_type == self.CLASS_CYCLIST:
            self.emb = tf.keras.layers.LSTM(
                units = 64,
                dropout=0.2,
                activation="tanh",
                kernel_initializer='glorot_uniform',
                return_sequences=self.return_sequences,
                name=self.name
            )
        elif self.embedding_type == self.FEATURE_HIST_TRAJ:
            self.emb = tf.keras.layers.LSTM(
                units = 64,
                dropout=0.2,
                activation="tanh",
                kernel_initializer='glorot_uniform',
                return_sequences=self.return_sequences,
                name=self.name
            )
        elif self.embedding_type == self.FEATURE_LANE_DEV:
            self.emb = tf.keras.layers.LSTM(
                units = 64,
                dropout=0.2,
                activation="tanh",
                kernel_initializer='glorot_uniform',
                return_sequences=self.return_sequences,
                name=self.name
            ) 
        elif self.embedding_type == self.FEATURE_LANE_GEO:
            self.emb = tf.keras.layers.LSTM(
                units = 64,
                dropout=0.2,
                activation="tanh",
                kernel_initializer='glorot_uniform',
                return_sequences=self.return_sequences,
                name=self.name
            )
        elif self.embedding_type == self.FEATURE_ROAD_GEO:
            self.emb = tf.keras.layers.Conv2D(
                filter=3,
                kernel_size=3,
                activation='relu',
                padding="same",
                name=self.name
            )       
        

        if self.dropout > 0:
            self.drop_l = tf.keras.layers.Dropout(self.dropout)


    def call(self, inputs, training=False):

        if self.emb is None:
            raise RuntimeError("layer not initialized!")

        out = self.emb(inputs)

        if self.drop_l is not None:
            out = self.drop_l(out, training=training)

        return out
    
    