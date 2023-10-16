
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import copy
from copy import deepcopy

from typing import Dict, NoReturn, List, Tuple, Union, Any

from models.layers.embedding import FeaturesEmbedding
from models.layers.graph_attention_network import GraphAttentionNetwork
from utils.weighted_categorical_crossentropy import weighted_categorical_crossentropy

"""
Maneuver Prediction using Graph Attention Network

- input:
    traj: trajectory of the vehicle (x,y)
    lane: lane geometry (x,y)
    deviation: lane deviation between the trajectory
                    and the lane geometry (d)
                    
    nodes: historical trajectory of neighbors (1, d)
    edges: connections within a interaction range
      
- output:
    maneuver probabilities
        + Lateral Maneuver
        - [LLC] left lane-change 
        - [RLC] right lane-change
        - [TL]  left turn
        - [TR]  right turn
        - [LK]  lane-keeping / straight at intersection
        
        + Longitudinal Maneuver
        - [ST]  stop
        - [ACC] acceleration
        - [DEC] deceleration
        - [KS]  keep speed
"""

class ManeuverPrediction(tf.keras.Model):
    
    def __init__(
        self,
        name:str,
        output_size:Dict,
        input_size:Dict,
        class_weights:List[float], 
        lateral_weights:List[float],
        longitudinal_weights:List[float],
        decoder_params:dict,
        **kwargs
    ):
        
        super(ManeuverPrediction, self).__init__(name=name, **kwargs)
              
        self.output_size=output_size
        self.class_weights=class_weights
        self.input_size=input_size
        self.lateral_weights = lateral_weights
        self.longitudinal_weights = longitudinal_weights

        self.decoder_params = decoder_params
        self.build([
            self.input_size['trajectory'],
            self.input_size['lane_geometry'],
            self.input_size['lane_deviation'],
        ])

        self.loss_tracker =\
            tf.keras.metrics.Mean(name='maneuver_loss')
        self.lat_loss_tracker =\
            tf.keras.metrics.Mean(name='lateral_loss')
        self.lon_loss_tracker =\
            tf.keras.metrics.Mean(name='longitudinal_loss')
    
        self.custom_metrics = self.create_metrics()

        # self.lateral_loss_fn =\
        #      weighted_categorical_crossentropy(self.lateral_weights)
        # self.longitudinal_loss_fn =\
        #      weighted_categorical_crossentropy(self.longitudinal_weights)

        self.compile(optimizer=tf.keras.optimizers.Adam(1e-4), run_eagerly=True)

    def create_metrics(self):
        metrics = {}

        for c in ['lateral', 'longitudinal']:
            metrics[c]={}
            metrics[c]['AUC']=\
                tf.keras.metrics.AUC()
            metrics[c]['crossentropy']=\
                tf.keras.metrics.CategoricalCrossentropy()
            metrics[c]['accuracy']=\
                tf.keras.metrics.CategoricalAccuracy()
            metrics[c]['recall']=\
                tf.keras.metrics.Recall()
            metrics[c]['precision']=\
                tf.keras.metrics.Precision()
                
        return metrics  

    def change_decoder(self, decoder_params:Dict)->Any:

        self.decoder_params = decoder_params

        self.__build_decoder(
            units=self.decoder_params['units'],
            drop_rate=self.decoder_params['drop_rate']
        )

        return self

    def __build_decoder(self, units:List[int], drop_rate:List[float]):
        if len(units) != len(drop_rate):
            raise ValueError(("Mismatch betwenn the number of dense layers (units)"
                                " and dropout layers (drop_rate)"))

        self.decoder = {}
        for maneuver in ['lateral', 'longitudinal']:
            self.decoder[maneuver] = {}
            self.decoder[maneuver]['attention']=\
                tf.keras.layers.MultiHeadAttention(
                    num_heads=4,
                    key_dim=2,
                    attention_axes=(1),
                    name=f"attention_{maneuver}"
                )
            self.decoder[maneuver]['dense']=\
                [
                    tf.keras.layers.Dense(
                        u, 
                        activation=tf.nn.leaky_relu, 
                        kernel_initializer='he_uniform',
                        name=f"dense_{u}_{maneuver}"
                    )
                    for u in units
                ]
            self.decoder[maneuver]['dropout']=\
                [
                    tf.keras.layers.Dropout(
                        prop, 
                        name=f"dropout_{prop}_{maneuver}"
                    )
                    for prop in drop_rate
                ]
            self.decoder[maneuver]["output"]=\
                tf.keras.layers.Dense(
                    self.output_size[f"{maneuver}_maneuver"],
                    activation="softmax",
                    name=f"{maneuver}_maneuver"
                )
 
    def build(self, input_shape):

        #norm and embedding
        _names = ['traj', 'lane_geo', 'lane_dev', 'surr']
        _types = [FeaturesEmbedding.FEATURE_HIST_TRAJ, 
                  FeaturesEmbedding.FEATURE_LANE_GEO,
                  FeaturesEmbedding.FEATURE_LANE_DEV,
                  FeaturesEmbedding.CLASS_SR_VEHICLE]

        self.norm = {
            name: tf.keras.layers.BatchNormalization(
                        axis=1,
                        name=f"norm_{name}"
                  )
            for name in _names
        }

        self.norm_decoder =\
            tf.keras.layers.BatchNormalization(
                        axis=-1,
                        name=f"norm_decoder"
            )

        self.embd = {
            name: FeaturesEmbedding(
                        embedding_type=_type,
                        dropout=0.3,
                        return_sequences=True,
                        name=f"embd_{name}"
                  )
            for name, _type in zip(_names, _types) 
        }
            
        #encoder target vehicle
        self._concat = tf.keras.layers.Concatenate()
        self._flatten = tf.keras.layers.Flatten() 

        self.enc_bilstm = {
            name: tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(
                            units,
                            activation="tanh",
                            return_sequences=True,
                            dropout=0.3
                        ),
                        merge_mode="concat",
                        name=f"enc_bilstm_{name}"
                    )
            for name, units in zip(['in', 'out'], [256, 256])
        }
        
        
        #encoder interaction
        self.gat_flatten = tf.keras.layers.Flatten()
        self.gat_interaction =\
            GraphAttentionNetwork(
                name="interaction",
                hidden_units=1024,
                num_heads=4,
                num_layers=2
            )

        #decoder
        self.__build_decoder(
            units=self.decoder_params['units'],
            drop_rate=self.decoder_params['drop_rate']
        )
        
        self.built=True

    def call(self, inputs, training=False):

        #########
        #inputs
        #########
        x_traj = inputs[0]
        x_lane_geo = inputs[1] 
        x_lane_dev = inputs[2]
        
        graph_edges = inputs[3]
        graph_nodes = inputs[4]

        #padding
        if x_lane_geo.shape[1] > x_traj.shape[1]:
            x_traj =\
                tf.keras.preprocessing.sequence.pad_sequences(
                    x_traj, 
                    maxlen=x_lane_geo.shape[1], 
                    padding='post',
                    value=0.0,
                    dtype=np.float32
                )
        if x_lane_geo.shape[1] > x_lane_dev.shape[1]:
            x_lane_dev =\
                tf.keras.preprocessing.sequence.pad_sequences(
                    x_lane_dev, 
                    maxlen=x_lane_geo.shape[1], 
                    padding='post',
                    value=0.0,
                    dtype=np.float32
                )

        #########        
        #GAT (interaction) 
        #########             
        nodes  = []
        edges  = []
        
        for n, e in zip(graph_nodes, graph_edges):
            #target
            x_tgt_traj =\
                self.norm["traj"](
                    tf.expand_dims(n[0].to_tensor(), axis=0),
                    training=training
                )
            x_tgt_traj =\
                self.embd["traj"](
                    x_tgt_traj,
                    training=training
                )
            
            x_tgt_traj = self._flatten(x_tgt_traj)

            #surroundings
            if n.nrows() > 1:
                x_tgt_surr =\
                    self.norm["surr"](
                        n[1:].to_tensor(),
                        training=training
                    )
                x_tgt_surr =\
                    self.embd["surr"](
                        x_tgt_surr, 
                        training=training
                    )
                
                x_tgt_surr = self._flatten(x_tgt_surr)
                        
                x_emb_nodes = tf.concat([x_tgt_traj, x_tgt_surr], axis=0)

            else:
                x_emb_nodes = x_tgt_traj


            nodes.append(x_emb_nodes)            
            edges.append(e.to_tensor())
        
        # del graph_nodes
        # del graph_edges

        #interaction features extraction
        interaction = self.gat_interaction([nodes, edges])
        tgt_interaction = tf.stack([inter[0] for inter in interaction])
        interaction = [tf.reduce_mean(inter, axis=0)
                        for inter in interaction]
        
        # del nodes
        # del edges
        #############
        #ego vehicle 
        ############# 

        ego = {}
        for name, feature in zip(['traj', 'lane_geo', 'lane_dev'],
                                 [x_traj, x_lane_geo, x_lane_dev]):
            ego[name] = self.norm[name](
                            feature,
                            training=training
                        )
            ego[name] = self.embd[name](
                            ego[name],
                            training=training
                        )
        enc = self._concat([feature for _, feature in ego.items()])

        for name in ['in', 'out']:
            enc = self.enc_bilstm[name](enc, training=training)
        enc = self._flatten(enc)
        
        # del x_traj
        # del x_lane_geo
        # del x_lane_dev

        #fusion
        enc = self._concat([enc, tgt_interaction, interaction])
        enc = self.norm_decoder(enc, training=training)
        
        out = {}

        for maneuver in ["lateral", "longitudinal"]:
            m = self.decoder[maneuver]["attention"](
                    enc, enc, training=training
                )
            for dense, drop in zip(self.decoder[maneuver]["dense"],
                                   self.decoder[maneuver]["dropout"]):
                m = dense(m)
                m = drop(m, training=training)
            out[maneuver] = self.decoder[maneuver]["output"](m)
        
        return out["lateral"], out["longitudinal"]

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weights=None):
        del x #we dont use x

        lat_pred  = y_pred[0]
        lon_pred  = y_pred[1]

        lat_true  = y[0]
        lon_true  = y[1]

        #LATERAL loss
        lat =\
            tf.keras.losses.CategoricalCrossentropy()(
                y_true=lat_true, 
                y_pred=lat_pred
            )
        # lat =\
        #     self.lateral_loss_fn(
        #         y_true=lat_true,
        #         y_pred=lat_pred
        #     ) 
        lat = tf.math.reduce_mean(lat)
        
        #LONGITUDINAL loss
        lon =\
            tf.keras.losses.CategoricalCrossentropy()(
                y_true=lon_true, 
                y_pred=lon_pred
            )
        # lon =\
        #     self.longitudinal_loss_fn(
        #         y_true=lon_true,
        #         y_pred=lon_pred
        #     )
        lon = tf.math.reduce_mean(lon)

        #intention
        loss =tf.math.reduce_sum(
                    [lat*self.class_weights[0], 
                     lon*self.class_weights[1]]
              )
        loss=loss/(self.class_weights[0]+self.class_weights[1])
        
        self.loss_tracker.update_state(loss)
        
        self.lat_loss_tracker.update_state(lat)
        self.lon_loss_tracker.update_state(lon)

        return loss

    def reset_metrics(self):
        self.loss_tracker.reset_states()
        self.lat_loss_tracker.reset_states()
        self.lon_loss_tracker.reset_states()

        for m in self.custom_metrics['lateral'].keys():
            self.custom_metrics['lateral'][m].reset_states()
            self.custom_metrics['longitudinal'][m].reset_states()

    @property
    def metrics(self):
        metrics = [
            self.loss_tracker,
            self.lat_loss_tracker,
            self.lon_loss_tracker
        ]

        return metrics

    def compute_metrics(self, x=None, y=None, y_pred=None, sample_weight=None):
       
        # metric_results =\
        #    super(ManeuverPrediction, self).compute_metrics(
        #        x, y, y_pred, sample_weight
        #    )
        del x #we dont use x
        metric_results = {}
        
        y_true_lat = y[0]
        y_true_lon = y[1]

        y_pred_lat = y_pred[0]
        y_pred_lon = y_pred[1]

        for m in self.custom_metrics['lateral'].keys():
            self.custom_metrics['lateral'][m].update_state(
                y_true_lat, y_pred_lat, sample_weight
            )
            self.custom_metrics['longitudinal'][m].update_state(
                y_true_lon, y_pred_lon, sample_weight
            )
        
        # metric_results['lateral'] = {}
        # metric_results['longitudinal'] = {}

        # for m in self.custom_metrics['lateral'].keys():
        #     metric_results['lateral'][m] =\
        #         self.custom_metrics['lateral'][m].result()
        #     metric_results['longitudinal'][m] =\
        #         self.custom_metrics['longitudinal'][m].result()
        
        # metric_results['lateral']['loss']=\
        #     self.lat_loss_tracker.result()
        
        metric_results['loss']=\
            self.loss_tracker.result()
        
        metric_results['lateral_loss']=\
            self.lat_loss_tracker.result()
        
        # metric_results['longitudinal']['loss']=\
        #     self.lon_loss_tracker.result()
        metric_results['longitudinal_loss']=\
            self.lon_loss_tracker.result()
        
                
        for m in self.custom_metrics['lateral'].keys():
            metric_results[f'lateral_{m}'] =\
                self.custom_metrics['lateral'][m].result()
            metric_results[f'longitudinal_{m}'] =\
                self.custom_metrics['longitudinal'][m].result()

        

        return metric_results

    # @tf.function
    def train_step(self, data):
        x = data[0]
        y_true = data[1]

        with tf.GradientTape() as tape: 

            #intention 
            lat_pred, lon_pred =\
                self(
                    [x[0], x[1], x[2], x[3], x[4]],
                     training=True
                )
            
            loss_intent =\
                self.compute_loss(
                    y=[y_true[0], y_true[1]], 
                    y_pred=[lat_pred, lon_pred]
                )

            metrics_intent=\
                self.compute_metrics(
                    y=[y_true[0], y_true[1]], 
                    y_pred=[lat_pred, lon_pred]
                )

        grads_intention =\
            tape.gradient(
                loss_intent, self.trainable_weights
            )
        # grads_intention =\
        #    [(tf.clip_by_norm(grad, clip_norm=10.0)) 
        #     for grad in grads_intention]

       
        #optimization
        self.optimizer.apply_gradients(
            zip(grads_intention, self.trainable_weights)
        )
               
        return metrics_intent

    # @tf.function
    def test_step(self, data):
        x = data[0]
        y_true = data[1]

        #maneuver intention
        lat_pred, lon_pred =\
            self([x[0], x[1], x[2], x[3], x[4]])
                    
        loss =\
            self.compute_loss(
                y=[y_true[0], y_true[1]], 
                y_pred=[lat_pred, lon_pred]
            )
        metrics=\
            self.compute_metrics(
                y=[y_true[0], y_true[1]], 
                y_pred=[lat_pred, lon_pred]
            )

        return metrics

    
    def evaluate(self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=False,
        **kwargs,
    ):
        x = deepcopy(x)

        if isinstance(x, Tuple):
            x = list(x)

        if not isinstance(x[3], tf.RaggedTensor):
            x[3] = tf.ragged.constant(x[3])
        if not isinstance(x[4], tf.RaggedTensor):
            x[4] = tf.ragged.constant(x[4])

        return super(ManeuverPrediction, self).evaluate(
               x=x,
               y=y,
               batch_size=batch_size,
               verbose=verbose,
               sample_weight=sample_weight,
               steps=steps,
               callbacks=callbacks,
               max_queue_size=max_queue_size,
               workers=workers,
               use_multiprocessing=use_multiprocessing,
               return_dict=return_dict,
               **kwargs
           )

    def predict(
        self,
        x,
        batch_size=None,
        verbose="auto",
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        x = deepcopy(x)
        
        if isinstance(x, Tuple):
            x = list(x)

        if not isinstance(x[3], tf.RaggedTensor):
            x[3] = tf.ragged.constant(x[3])
        if not isinstance(x[4], tf.RaggedTensor):
            x[4] = tf.ragged.constant(x[4])
        
        return super(ManeuverPrediction, self).predict(
               x=x,
               batch_size=batch_size,
               verbose=verbose,
               steps=steps,
               callbacks=callbacks,
               max_queue_size=max_queue_size,
               workers=workers,
               use_multiprocessing=use_multiprocessing
           )