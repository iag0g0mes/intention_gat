import os
import sys
import argparse 
import configparser
import copy
import json
import datetime
import shutil

import numpy as np
import pandas as pd
import pickle as pkl

from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, Tuple, List

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from ast import literal_eval as make_tuple
from keras.engine import data_adapter
from keras import callbacks as callbacks_module
from keras.utils import tf_utils
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

from models.layers.graph_maneuver import ManeuverPrediction
from utils.preprocessing import load_data, preprocessing
from utils.train_test_split import split_data

def parse_arguments() -> Any:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Directory where the config file is saved",
    )

    return parser.parse_args()

def load_kfold_data(
    train_dir:str, 
    kfold:int, 
    input_shape:Dict,
    test_rate:float=0.2
):
    folds_path = [os.path.join(train_dir, d) 
                    for d in os.listdir(train_dir) 
                    if d.startswith("fold")]
    if len(folds_path) > kfold:
        folds_path = np.array(folds_path)[:kfold]

    train_data = {}

    for fp in folds_path:
        _data = load_data(fp)#, size=100) 
        
        for key, value in _data.items():
            if key in train_data:
                if isinstance(value, List):
                    train_data[key] += value
                elif isinstance(value, np.ndarray):
                    train_data[key] = np.append(train_data[key], value, axis=0)
                elif isinstance(value, pd.DataFrame):
                    train_data[key] = pd.concat([train_data[key], value], ignore_index=True, axis=0)
            else:
                train_data[key]=value

    x_train, y_train = preprocessing(train_data, input_shape)
    x_val, y_val = None, None

    if test_rate > 0:
        (x_train, y_train, _),\
        (x_val, y_val, _),\
        (_,_, _)=\
            split_data(x_train, y_train, test_rate=0.0, val_rate=test_rate)
    
    return (x_train, y_train),(x_val, y_val)
        
    

def print_shape(mode, x, y):
    head = "\033[94m[Basic Model][Graph Maneuver Prediction][main]\033[0m"
    
    x_names = ['historical_trajectory', 
               'lane_geometry', 
               'lane_deviation', 
               'edges', 
               'nodes_state']
    y_names = ['lateral_intention',
               'longitudinal_intention']
    
    print(f'{head}[dataset] {mode}:')
    for idx, value in enumerate(zip(x, x_names)):
        if isinstance(value[0], List):
            shape = len(value[0])
        else:
            shape = np.shape(value[0])
        # shape = np.shape(value[0]) if idx <3 else len(value[0])
        print(f'\t{value[1]}:{shape}')
    for value, name in zip(y, y_names):
        print(f'\t{name}:{np.shape(value)}')

def train(
    model:ManeuverPrediction,
    x_train:np.ndarray,
    y_train:np.ndarray,
    x_val:np.ndarray,
    y_val:np.ndarray,
    log_dir:str,
    epochs:int,
    batch_size:int, 
) -> ManeuverPrediction:
       
    #datasets
    x_train[3] = tf.ragged.constant(x_train[3])
    x_train[4] = tf.ragged.constant(x_train[4])
    
    x_val[3] = tf.ragged.constant(x_val[3])
    x_val[4] = tf.ragged.constant(x_val[4])
    
    train_dataset = data_adapter.get_data_handler(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        workers=8,
        use_multiprocessing=True,
        model=model,
    )

    model._eval_data_handler = data_adapter.get_data_handler(
        x=x_val,
        y=y_val,
        batch_size=batch_size,
        epochs=1,
        shuffle=True,
        workers=8,
        use_multiprocessing=True,
        model=model,
    )

    #callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-5, 
        patience=15,
        verbose=1,
        mode="min",
        restore_best_weights=True
    )  
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(log_dir, 'model'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1,
    )
            
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    )

    callbacks =\
        callbacks_module.CallbackList(
            [early_stop, checkpoint, tensorboard_callback],
            add_history=True,
            add_progbar=1,
            model=model,
            verbose=1,
            epochs=epochs,
            steps=train_dataset.inferred_steps
        )
    
    #training
    callbacks.on_train_begin()
    training_logs = None
    logs = None
    
    for epoch, iterator in train_dataset.enumerate_epochs():
        model.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        
        with train_dataset.catch_stop_iteration():
            for step in train_dataset.steps():
                with tf.profiler.experimental.Trace(
                    'train',
                    epoch_num=epoch,
                    step_num=step,
                    batch_size=batch_size,
                    _r=1):
                    
                    callbacks.on_train_batch_begin(step)
                    
                    xi = next(iterator)
                    
                    tmp_logs = model.train_step(xi)
                                        
                    logs = tmp_logs
                    end_step = step + train_dataset.step_increment
                    
                    callbacks.on_train_batch_end(end_step, logs)
                    
                    if model.stop_training:
                        break
        
        logs = tf_utils.sync_to_numpy_or_python_type(logs)
        epoch_logs = copy.copy(logs)
    

        # callbacks.on_test_begin()
        # for _, val_iter in val_dataset.enumerate_epochs():  
        #     model.reset_metrics()
        #     with val_dataset.catch_stop_iteration():
        #         for step in val_dataset.steps():
        #             with tf.profiler.experimental.Trace(
        #                 "val", step_num=step, _r=1
        #             ):
        #                 callbacks.on_test_batch_begin(step)
        #                 tmp_logs = model.test_step(val_iter)
        #                 logs = tmp_logs
        #                 end_step = step + val_dataset.step_increment
        #                 callbacks.on_test_batch_end(end_step, logs)
        # logs = tf_utils.sync_to_numpy_or_python_type(logs)
        # callbacks.on_test_end(logs=logs)

                   
        val_logs = model.evaluate(
            x = x_val,
            y = y_val,
            batch_size=batch_size,
            callbacks=callbacks,
            workers=8,
            use_multiprocessing=True,
            return_dict=True,
            _use_cached_eval_dataset=True
        )
       
        val_logs = {'val_' + name: val for name, val in val_logs.items()}
        epoch_logs.update(val_logs)

        callbacks.on_epoch_end(epoch, epoch_logs)
        training_logs = epoch_logs
        
        if model.stop_training:
            break
                    
    if model._eval_data_handler is not None:
        del model._eval_data_handler
        
    callbacks.on_train_end(logs=training_logs)
         
    return model

if __name__ == "__main__":
    
    print("\033[94m[Basic Model][Graph Maneuver Prediction][main]\033[0m creating model...")
    
    args = parse_arguments()
    
    if not os.path.exists(args.cfg):
        raise FileNotFoundError(("config file not found!"
                                 f"(cfg:{args.cfg})"))

    config = configparser.ConfigParser()
    config.read(args.cfg)    

    for key, param in config["DIRS"].items():
        if key == 'weights_path':
            continue
        if not os.path.exists(param):
            raise FileNotFoundError((f"{key} not found!"
                                     f"({key}:{param})"))
                
    input_shape={}
    input_shape['trajectory'] =\
        make_tuple(config["INPUT"]["trajectory"])
    input_shape['lane_geometry'] =\
        make_tuple(config["INPUT"]["lane_geometry"])
    input_shape['lane_deviation'] =\
        make_tuple(config["INPUT"]["lane_deviation"])
    
    output_shape={}
    output_shape['lateral_maneuver'] =\
        config["OUTPUT"].getint("lateral")
    output_shape['longitudinal_maneuver'] =\
        config["OUTPUT"].getint("longitudinal")
    
    
    print("\033[94m[Basic Model][Graph Maneuver Prediction][main]\033[0m loading data...")
    
    #dir
    train_dir = config['DIRS']['train_dir']
    test_dir = config['DIRS']['test_dir']
    model_dir = config['DIRS']['model_dir']
    
    #train param
    val_rate = config['PARAMS'].getfloat('val_rate')
    batch_size = config['PARAMS'].getint('batch_size')
    epochs = config['PARAMS'].getint('epochs')
    kfold = config['PARAMS'].getint('kfold')

    #loss param
    class_weights = make_tuple(config['PARAMS']['class_weights'])
    lateral_weights = make_tuple(config['PARAMS']['lateral_weight'])
    longitudinal_weights = make_tuple(config['PARAMS']['longitudinal_weight'])

    #load data
    (x_train, y_train),\
    (x_val, y_val) =\
        load_kfold_data(train_dir, kfold, test_rate=val_rate, input_shape=input_shape)

    print_shape(mode='train', x=x_train, y=y_train)
    print_shape(mode='val', x=x_val, y=y_val)
    
    print("\033[94m[Basic Model][Graph Maneuver Prediction][main]\033[0m creating model...")
        
    decoder_params = dict()
    decoder_params['units']=make_tuple(config['DECODER']['units'])
    decoder_params['drop_rate']=make_tuple(config['DECODER']['dropout'])

    model = ManeuverPrediction(
                name="ManeuverPrediction",
                input_size=input_shape, 
                output_size=output_shape,
                class_weights=class_weights,
                lateral_weights=lateral_weights,
                longitudinal_weights=longitudinal_weights,
                decoder_params=decoder_params                
            )
    
    
    if config['PARAMS']['mode'] == "train":
        print("\033[94m[Basic Model][Graph Maneuver Prediction][main]\033[0m training model...")
        
        data_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/" + data_str
        log_dir = os.path.join(model_dir, log_dir)
        print(f"\033[94m[Basic Model][Graph Maneuver Prediction][main]\033[0m[train] log_dir:{log_dir}")
        
        
        model = train(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            log_dir=log_dir,
            epochs=epochs,
            batch_size=batch_size
        )

        history_file = os.path.join(log_dir, "history.json")
        with open(history_file, 'w') as f:
            json.dump(model.history.history, f)

        #params 
        params_file = os.path.join(log_dir, "basic_model.ini")
        params_source = os.path.join(FILE.parents[0], "cfg", "basic_model.ini")
        shutil.copy(params_source, params_file)

        #model (safecopy)
        model_file = os.path.join(log_dir, "graph_maneuver.py")
        model_source = os.path.join(FILE.parents[0], "layers", "graph_maneuver.py")
        shutil.copy(model_source, model_file)

    
    elif config['PARAMS']['mode']=="test":
        print("\033[94m[Basic Model][Graph Maneuver Prediction][main]\033[0m testing model...")

        weights_path = config['DIRS']['weights_path']

        # if not os.path.exists(weights_path):
        #     raise FileNotFoundError(f"Weight path not found! {weights_path}")

        log_dir = Path(weights_path).parents[0]

        _x_init = []
        for x in x_val:
            if isinstance(x, list):
                _x_init.append(tf.ragged.constant([x[0]]))
            else:
                _x_init.append(np.array([x[0]]))
        model(_x_init)
        model.load_weights(weights_path)

        test_data = load_data(test_dir)#, size=100)
        x_test, y_test = preprocessing(test_data, input_shape)
        del test_data
        
        print_shape(mode='test', x=x_test, y=y_test)
        
        print("\033[94m[Graph Maneuver Prediction][main]\033[0m evaluating model...")
        
        maneuvers_lat = ["LLC", "RLC", "TL", "TR", "LK"]
        maneuvers_lat = [p + m for p in ['true_', 'pred_'] for m in maneuvers_lat]

        maneuvers_lon = ["ST", "ACC", "DEC", "KS"]
        maneuvers_lon = [p + m for p in ['true_', 'pred_'] for m in maneuvers_lon]

        columns = maneuvers_lat + maneuvers_lon
        
        for name, x_true, y_true in zip(["train", "val", "test"],
                                        [x_train, x_val, x_test],
                                        [y_train, y_val, y_test]):
            eval_logs = model.evaluate(
                    x = x_true,
                    y = y_true,
                    batch_size=1,
                    workers=4,
                    verbose=1,
                    use_multiprocessing=True,
                    return_dict=True,
                )
            
            eval_file = os.path.join(log_dir, f"{name}_eval.json")
            with open(eval_file, 'w') as f:
                json.dump(eval_logs, f)
            
            print(f"\t {name}:")
            for metric, value in eval_logs.items():
                print("\t\t {}={:.3f}".format(metric, value))
            

            y_pred_file = os.path.join(log_dir, f"{name}_predict.csv")
            lat_pred, lon_pred = model.predict(x_true)
            data_conct = np.concatenate([y_true[0], lat_pred, y_true[1], lon_pred], axis=1)
            pd.DataFrame(data_conct, columns=columns).to_csv(y_pred_file)

        print("\033[94m[Graph Maneuver Prediction][main]\033[0m DONE!")