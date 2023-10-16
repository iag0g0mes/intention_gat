import os
from hydra import initialize_config_dir
import numpy as np
import pandas as pd

import tensorflow as tf

from typing import Any

"""
[Generic]

Graph Attention Layer and Graph Multi-head Attention Layer

Font: https://keras.io/examples/graph/gat_node_classification/
"""
class GraphAttention(tf.keras.layers.Layer):
    
    def __init__(
        self, 
        units:int, 
        kernel_initializer:str="glorot_uniform",
        kernel_regularizer:str="l2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.units = units
        
        self.kernel_initializer =\
            tf.keras.initializers.get(kernel_initializer)
            
        self.kernel_regularizer =\
            tf.keras.regularizers.get(kernel_regularizer)
            
    def build(self, input_shape):
        
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1][1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        
        self.kernel_attention = self.add_weight(
            shape=(self.units*2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
             regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        
        self.built=True
        
    def call(self, inputs):
        #TODO: convert inputs to ragged tensor
        #TODO: replace generators by ragged operators
        # obs: is using generators faster than ragged tensors? 
        #      - so far is faster (test it more...)
        #TODO: change list of edges to adjacency matrix
        
        node_states, edges = inputs
        
        edges = [tf.cast(ed, dtype=tf.int32) for ed in edges]

        # Linearly transform node states
        #node_states_transformed = tf.matmul(node_states, self.kernel)
        node_states_transformed = [tf.matmul(nodes, self.kernel)
                                   for nodes in node_states]
        
        # (1) Compute pair-wise attention scores
        node_states_expanded =\
            [
                tf.gather(nodes, nodes_edges)
                    for nodes, nodes_edges in 
                    zip(node_states_transformed, edges)
            ]
        
        node_states_expanded = \
            [
                tf.reshape(nodes, (tf.shape(nodes_edges)[0], -1))
                    for nodes, nodes_edges in 
                    zip(node_states_expanded, edges)
            ]
        
        
        attention_scores = \
            [   
                tf.nn.leaky_relu(
                    tf.matmul(nodes, self.kernel_attention)
                ) for nodes in node_states_expanded
            ]
        
        attention_scores = [tf.squeeze(att, -1) 
                            for att in attention_scores]
        
        
        
        # (2) Normalize attention scores
        attention_scores = [tf.math.exp(tf.clip_by_value(att, -2, 2))
                            for att in attention_scores]
        
        
        attention_scores_sum = \
            [
                tf.math.unsorted_segment_sum(
                    data=att,
                    segment_ids=nodes_edges[:, 0],
                    num_segments=tf.reduce_max(nodes_edges[:, 0]) + 1,
                ) for att, nodes_edges in 
                    zip(attention_scores, edges)
            ]
        
        
        attention_scores_sum = \
            [
                tf.repeat(
                    att, 
                    tf.math.bincount(tf.cast(nodes_edges[:, 0], "int32"))
                ) for att, nodes_edges in 
                    zip(attention_scores_sum, edges)
            ]
        
        attention_scores_norm =\
            [
                att/att_sum
                for att, att_sum in 
                zip(attention_scores,attention_scores_sum)
            ]
        

        
        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = \
            [
                tf.gather(nodes, nodes_edges[:, 1])
                for nodes, nodes_edges in
                zip(node_states_transformed, edges)
            ]
        
        
        out = \
            [
                tf.math.unsorted_segment_sum(
                    data= nodes* att[:, tf.newaxis],
                    segment_ids=nodes_edges[:, 0],
                    num_segments=tf.shape(n_state)[0],
                )for nodes, att, nodes_edges, n_state in 
                 zip(node_states_neighbors, 
                     attention_scores_norm, 
                     edges, 
                     node_states)
            ]
        return out
    
class MultiHeadGraphAttention(tf.keras.layers.Layer):
    
    def __init__(
        self,
        units:int, 
        num_heads:int=8,
        merge_type:str="concat",
        kernel_initializer:str="glorot_uniform",
        kernel_regularizer:str="l2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.units = units
        
        self.attention_layers =\
            [GraphAttention(units) for _ in range(num_heads)]

        self.kernel_initializer =\
            tf.keras.initializers.get(kernel_initializer)
            
        self.kernel_regularizer =\
            tf.keras.regularizers.get(kernel_regularizer)
    
    def build(self, input_shape):
        
        _shape = (self.units*self.num_heads, self.units)\
                    if self.merge_type=="concat" else\
                        (self.units, self.units)

        self.kernel = self.add_weight(
            shape=_shape,
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )

        self.built = True

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        
        #aggregate outputs of the same sample
        outputs = \
            [
                [o[i] for o in outputs]
                for i in range(0, len(outputs[0]))
            ]
        
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            
            outputs = \
                [
                    tf.concat(out, axis=-1)
                    for out in outputs 
                ]
        else:
            outputs = \
                [
                    tf.reduce_mean(tf.stack(out, axis=-1), axis=-1)
                    for out in outputs
                ]
                
        #output
        outputs = [tf.matmul(out, self.kernel)
                    for out in outputs]

        # Activate and return node states
        outputs = \
            [
                tf.nn.leaky_relu(out)
                for out in outputs
            ]
            
        return outputs

class GraphAttentionNetwork(tf.keras.layers.Layer):
    def __init__(
        self,
        name:str,
        hidden_units,
        num_heads,
        num_layers,
        kernel_initializer:str="glorot_uniform",
        kernel_regularizer:str="l2",
        **kwargs,
    ):
        super(GraphAttentionNetwork, self).__init__(name=name,**kwargs)
        
        self.units = hidden_units

        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        
        self.kernel_initializer =\
            tf.keras.initializers.get(kernel_initializer)
            
        self.kernel_regularizer =\
            tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1][1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )

        self.built = True

    def call(self, inputs):
        node_states, edges = inputs
            
        #Linearly transform node states
        node_states_transformed = [tf.matmul(nodes, self.kernel)
                                   for nodes in node_states]

        x = node_states_transformed
        
        for attention_layer in self.attention_layers:
            out = attention_layer([x, edges]) 
                       
            x = [xi + oi 
                 for xi, oi in zip(x, out)]

        return x

    
