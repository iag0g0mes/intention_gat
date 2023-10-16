
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

from typing import Dict, NoReturn, List, Tuple, Union

from models.layers.embedding import FeaturesEmbedding

"""
Maneuver Prediction using Bidirectional LSTM

- input:
    traj: trajectory of the vehicle (x,y)
    lane: lane geometry (x,y)
    deviation: lane deviation between the trajectory
                    and the lane geometry (d)
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
        **kwargs
    ):
        
        super(ManeuverPrediction, self).__init__(name=name, **kwargs)
              
        self.output_size=output_size
        self.class_weights=class_weights
        self.input_size=input_size

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
    

    def build(self, input_shape):
        
        #input
        # self._in_traj = tf.keras.layers.Input(
        #                 shape=input_shape[0],
        #                 name="trajectory"
        #             )
        # self._in_lane = tf.keras.layers.Input(
        #                 shape=input_shape[1],
        #                 name="lane_geometry"
        #             )
        # self._in_dev = tf.keras.layers.Input(
        #                 shape=input_shape[2],
        #                 name="lane_deviation"
        #             )

        #norm and embedding
        self._norm_traj =\
            tf.keras.layers.BatchNormalization(axis=1)  
    
        self._norm_lane =\
            tf.keras.layers.BatchNormalization(axis=1)   
            
        self._norm_dev =\
            tf.keras.layers.BatchNormalization(axis=1)

        self._emb_traj =\
            FeaturesEmbedding(
                embedding_type=FeaturesEmbedding.FEATURE_HIST_TRAJ,
                dropout=0.1,
                return_sequences=True
            )

        self._emb_lane =\
            FeaturesEmbedding(
                embedding_type=FeaturesEmbedding.FEATURE_LANE_GEO,
                dropout=0.1,
                return_sequences=True
            )

        self._emb_dev =\
            FeaturesEmbedding(
                embedding_type=FeaturesEmbedding.FEATURE_LANE_DEV,
                dropout=0.1,
                return_sequences=True
            )

        #encoder 
        self._concat = tf.keras.layers.Concatenate()
        self._flatten = tf.keras.layers.Flatten() 

        self._enc_bidirectional_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                256,
                activation="tanh",
                return_sequences=True,
                dropout=0.2
            ),
            merge_mode="concat",
        )
        
        self._out_bidirectional_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                256,
                activation="tanh",
                return_sequences=True,
                dropout=0.2
            ),
            merge_mode="concat"
        )
        
        #decoder lat
        #self._att_lat = tf.keras.layers.Attention(
        #    use_scale=True,
        #    name='att_intention_lat'
        #)
        self._att_lat =\
            tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=2,
                attention_axes=(1)
            )
        
        self._dense_lat =[
            tf.keras.layers.Dense(
                units, 
                activation=tf.nn.leaky_relu, 
                kernel_initializer='he_uniform'
            )
            for units in [1024, 512, 256]
        ]

        self._drop_lat=[
            tf.keras.layers.Dropout(prop)
            for prop in [0.2, 0.2, 0.1]
        ]
                
        self._dec_out_lat = tf.keras.layers.Dense(
            self.output_size['lateral_maneuver'],
            activation="softmax",
            name="lateral_maneuver"
        )
        
        #decoder lon
        #self._att_lon = tf.keras.layers.Attention(
        #    use_scale=True,
        #    name='att_intention_lon'
        #)
        self._att_lon =\
            tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=2,
                attention_axes=(1)
            )
        
        self._dense_lon =[
            tf.keras.layers.Dense(
                units, 
                activation=tf.nn.leaky_relu,  
                kernel_initializer='he_uniform'
            )
            for units in [1024, 512, 256]
        ]

        self._drop_lon=[
            tf.keras.layers.Dropout(prop)
            for prop in [0.2, 0.2, 0.1]
        ]

        self._dec_out_lon = tf.keras.layers.Dense(
            self.output_size['longitudinal_maneuver'],
            activation="softmax",
            name="longitudinal_maneuver"
        )
        
        self.built=True

    def call(self, inputs, training=False):
        #inputs
        x_traj = inputs[0]
        x_lane_geo = inputs[1] 
        x_lane_dev = inputs[2]
                
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
        
        # x_traj = self._in_traj(x_traj)
        # x_lane_geo = self._in_lane(x_lane_geo)
        # x_lane_dev = self._in_dev(x_lane_dev)

        #normalization        
        x_traj = self._norm_traj(
                    x_traj, 
                    training=training
                )
        
        x_lane_geo = self._norm_lane(
                        x_lane_geo, 
                        training=training
                    )
        x_lane_dev = self._norm_dev(
                        x_lane_dev, 
                        training=training
                    )     
    
        #embedding
        x_traj = self._emb_traj(
                    x_traj,
                    training=training
                )
        x_lane_geo = self._emb_lane(
                        x_lane_geo, 
                        training=training
                    )
        x_lane_dev = self._emb_dev(
                        x_lane_dev,  
                        training=training
                    )

        #concat   
        x = self._concat([x_traj, x_lane_geo, x_lane_dev])
            
        #encoder
        x = self._enc_bidirectional_lstm(x)
        enc = self._out_bidirectional_lstm(x)
        enc = self._flatten(enc)
        
        #lat
        #xlat = self._att_lat([enc,enc])
        xlat = self._att_lat(enc, enc)
        for dense, drop  in zip(self._dense_lat, self._drop_lat):
            xlat = dense(xlat)
            xlat = drop(xlat, training=training)
        out_lat = self._dec_out_lat(xlat)

        #lon
        #xlon = self._att_lon([enc,enc])
        xlon = self._att_lon(enc, enc)
        for dense, drop  in zip(self._dense_lon, self._drop_lon):
            xlon = dense(xlon)
            xlon = drop(xlon, training=training)        
        out_lon = self._dec_out_lon(xlon)

        return out_lat, out_lon

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
        
        #LONGITUDINAL loss
        lon =\
            tf.keras.losses.CategoricalCrossentropy()(
                y_true=lon_true, 
                y_pred=lon_pred
            )

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


    def train_step(self, data):
        x = data[0]
        y_true = data[1]
                   
        with tf.GradientTape() as tape: 

            #intention 
            lat_pred, lon_pred =\
                self(
                    [x[0], x[1], x[2]],
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
        grads_intention =\
           [(tf.clip_by_norm(grad, clip_norm=10.0)) 
            for grad in grads_intention]

       
        #optimization
        self.optimizer.apply_gradients(
            zip(grads_intention, self.trainable_weights)
        )
               
        return metrics_intent

    def test_step(self, data):
        x = data[0]
        y_true = data[1]
        
        #maneuver intention
        lat_pred, lon_pred =\
            self([x[0], x[1], x[2]])
                    
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