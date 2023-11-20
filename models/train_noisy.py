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
from utils.preprocessing import load_data, preprocessing, print_shape
from utils.train_test_split import split_data
from utils.pseudo_labels import get_pseudo_label
from utils.data_augmentation import data_augmentation

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
    test_rate:float=0.2,
    size:int=-1
):
    folds_path = [os.path.join(train_dir, d) 
                    for d in os.listdir(train_dir) 
                    if d.startswith("fold")]
    if len(folds_path) > kfold:
        folds_path = np.array(folds_path)[:kfold]

    train_data = {}

    for fp in folds_path:
        _data = load_data(fp, size=size) 
        
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

    x_train, y_train, seqs_train = preprocessing(train_data, input_shape, return_sequence=True)
    x_val, y_val, seqs_val = None, None, None

    if test_rate > 0:
        (x_train, y_train, seqs_train),\
        (x_val, y_val, seqs_val),\
        (_,_, _)=\
            split_data(x_train, y_train, seqs=seqs_train, test_rate=0.0, val_rate=test_rate)
    
    return (x_train, y_train, seqs_train),(x_val, y_val, seqs_val)
        
    

def train(
    model:ManeuverPrediction,
    x_labeled:Tuple[np.ndarray, np.ndarray, np.ndarray, List, List],
    y_labeled:Tuple[np.ndarray,np.ndarray],
    x_unlabeled:Tuple[np.ndarray, np.ndarray, np.ndarray, List, List],
    y_unlabeled:Tuple[np.ndarray,np.ndarray],
    x_val:Tuple[np.ndarray, np.ndarray, np.ndarray, List, List],
    y_val:Tuple[np.ndarray,np.ndarray],
    log_dir:str,
    epochs:int,
    batch_size:int, 
) -> ManeuverPrediction:
       
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), run_eagerly=True)
           
    #datasets
    x_train = [None]*5
    x_train[0] = np.vstack([x_labeled[0], x_unlabeled[0]])
    x_train[1] = np.vstack([x_labeled[1], x_unlabeled[1]])
    x_train[2] = np.vstack([x_labeled[2], x_unlabeled[2]])
    x_train[3] = x_labeled[3] + x_unlabeled[3]
    x_train[4] = x_labeled[4] + x_unlabeled[4]

    x_train[3] = tf.ragged.constant(x_train[3])
    x_train[4] = tf.ragged.constant(x_train[4])

    y_train = [None]*2
    y_train[0] = np.vstack([y_labeled[0], y_unlabeled[0]])
    y_train[1] = np.vstack([y_labeled[1], y_unlabeled[1]])

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
        min_delta=1e-4, 
        patience=10,
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

def noisy_student_step(
    teacher:ManeuverPrediction,
    X_labeled:Tuple,
    Y_labeled:Tuple,
    X_unlabeled:Tuple,
    X_val:Tuple,
    Y_val:Tuple,
    log_dir:str,
    augmentation_opt:Tuple[str],
    augmentation_prob:Tuple[float],
    decoder_student :dict,
    epochs:int=100,
    batch_size:int=16,
    samples_per_class:int=10000,
    pseudo_label_mode:str="soft",
    pseudo_label_threshold:float=0.5
)->ManeuverPrediction:

    print("\033[94m[Noisy Student][Graph Maneuver Prediction][noisy_student_step]\033[0m getting pseudo-labels")


    x_pseudo, y_pseudo=\
        get_pseudo_label(
            model=teacher,
            X=X_unlabeled,
            mode=pseudo_label_mode,
            threshold=pseudo_label_threshold
        )
    print_shape(mode='pseudo_label', x=x_pseudo, y=y_pseudo)

    print("\033[94m[Noisy Student][Graph Maneuver Prediction][noisy_student_step]\033[0m getting augmented data")
    print(f"\tdata_augmentation: {augmentation_opt} | {augmentation_prob}")
    print(f"\tsamples_per_class: {samples_per_class}")
    x_pseudo, y_pseudo=\
        data_augmentation(
            X=x_pseudo,
            Y=y_pseudo,
            operations=augmentation_opt, 
            opt_probs =augmentation_prob,
            samples_per_class=samples_per_class,
        )
    print_shape(mode='data_augmentation', x=x_pseudo, y=y_pseudo)

    print("\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m training model...")
    
    # student = ManeuverPrediction(
    #         name="ManeuverPrediction",
    #         input_size=input_shape, 
    #         output_size=output_shape,
    #         class_weights=class_weights,
    #         lateral_weights=lateral_weights,
    #         longitudinal_weights=longitudinal_weights,
    #         decoder_params=decoder_student                
    #     )

    student = teacher.change_decoder(decoder_student)
    del teacher

    student = train(
            model=student,
            x_labeled=X_labeled,
            y_labeled=Y_labeled,
            x_unlabeled=x_pseudo,
            y_unlabeled=y_pseudo,
            x_val=X_val,
            y_val=Y_val,
            log_dir=log_dir,
            epochs=epochs,
            batch_size=batch_size
        )

    history_file = os.path.join(log_dir, "history.json")
    with open(history_file, 'w') as f:
        json.dump(student.history.history, f)

    #params 
    params_file = os.path.join(log_dir, "noisy_student.ini")
    params_source = os.path.join(FILE.parents[0], "cfg", "noisy_student.ini")
    shutil.copy(params_source, params_file)

    #model (safecopy)
    model_file = os.path.join(log_dir, "graph_maneuver.py")
    model_source = os.path.join(FILE.parents[0], "layers", "graph_maneuver.py")
    shutil.copy(model_source, model_file)


    return student
    


if __name__ == "__main__":
    
    print("\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m creating model...")
    
    args = parse_arguments()
    
    if not os.path.exists(args.cfg):
        raise FileNotFoundError(("config file not found!"
                                 f"(cfg:{args.cfg})"))

    config = configparser.ConfigParser()
    config.read(args.cfg)    

    for key, param in config["DIRS"].items():
        if (key == 'teacher_dir') or (key=='weights_path'):
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
    
    
    print("\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m loading data...")
    
    #dir param
    labeled_dir = config['DIRS']['labeled_dir']
    unlabeled_dir = config['DIRS']['unlabeled_dir']
    test_dir = config['DIRS']['test_dir']
    teacher_dir = config['DIRS']['teacher_dir']
    model_dir = config['DIRS']['model_dir']
    
    #noisy param
    noisy_steps  = config['NOISY'].getint('steps')
    samples_per_class = config['NOISY'].getint('samples_per_class')
    augmentation_opt = make_tuple(config['NOISY']['data_augmentation'])
    augmentation_prob = make_tuple(config['NOISY']['augmentation_prob'])
    pseudo_label_mode = config['NOISY']['pseudo_label']
    pseudo_label_threshold = config['NOISY'].getfloat('threshold')

    if len(augmentation_opt) != len(augmentation_prob):
        raise ValueError(("Mismatch on the sizes of the augmentation operations"
                          " vector and their probabilities"))

    #train param
    val_rate = config['PARAMS'].getfloat('val_rate')
    batch_size = config['PARAMS'].getint('batch_size')
    epochs = config['PARAMS'].getint('epochs')
    kfold = config['PARAMS'].getint('kfold')
    size= config['PARAMS'].getint('size')

    #loss param
    class_weights = make_tuple(config['PARAMS']['class_weights'])
    lateral_weights = make_tuple(config['PARAMS']['lateral_weight'])
    longitudinal_weights = make_tuple(config['PARAMS']['longitudinal_weight'])

    #load data
    (x_labeled, y_labeled, seqs_labeled),\
    (x_val, y_val, seqs_val) =\
        load_kfold_data(labeled_dir, 
                        kfold, 
                        test_rate=val_rate, 
                        input_shape=input_shape,
                        size=size)

    unlabeled_data = load_data(unlabeled_dir, size=size)
    x_unlabeled, y_unlabeled, seqs_unlabeled = preprocessing(unlabeled_data, input_shape, return_sequence=True)
    del unlabeled_data

    print_shape(mode='train_labeled', x=x_labeled, y=y_labeled)
    print_shape(mode='train_unlabeled', x=x_unlabeled, y=None)
    print_shape(mode='val', x=x_val, y=y_val)
    
    print("\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m creating model...")
    
    decoder_teacher = dict()
    decoder_teacher['units']= make_tuple(config['TEACHER-DECODER']['units'])
    decoder_teacher['drop_rate']=make_tuple(config['TEACHER-DECODER']['dropout'])

    decoder_student = dict()
    decoder_student['units']= make_tuple(config['STUDENT-DECODER']['units'])
    decoder_student['drop_rate']= make_tuple(config['STUDENT-DECODER']['dropout'])

    

    if config['PARAMS']['mode'] == "train":
        print("\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m loading teacher model...")
        print(f"\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m teacher_dir:{teacher_dir}")
        
        teacher = ManeuverPrediction(
                name="ManeuverPrediction",
                input_size=input_shape, 
                output_size=output_shape,
                class_weights=class_weights,
                lateral_weights=lateral_weights,
                longitudinal_weights=longitudinal_weights,
                decoder_params=decoder_teacher                
            )

        _x_init = []
        for x in x_val:
            if isinstance(x, list):
                _x_init.append(tf.ragged.constant([x[0]]))
            else:
                _x_init.append(np.array([x[0]]))

        teacher(_x_init)
        teacher.load_weights(teacher_dir)
        del _x_init

        print("\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m training model...")
        
        data_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/" + data_str
        log_dir = os.path.join(model_dir, log_dir)
        print(f"\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m[train] log_dir:{log_dir}")
        
        teacher = noisy_student_step(
            teacher=teacher,
            X_labeled=x_labeled,
            Y_labeled=y_labeled,
            X_unlabeled=x_unlabeled,
            X_val=x_val,
            Y_val=y_val,
            log_dir=log_dir,
            augmentation_opt=augmentation_opt,
            augmentation_prob=augmentation_prob,
            epochs=epochs,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
            pseudo_label_mode=pseudo_label_mode,
            pseudo_label_threshold=pseudo_label_threshold,
            decoder_student=decoder_student
        )

    elif config['PARAMS']['mode'] == "test":
        print("\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m testing model...")
        weights_path = config['DIRS']['weights_path']

        model = ManeuverPrediction(
                name="ManeuverPrediction",
                input_size=input_shape, 
                output_size=output_shape,
                class_weights=class_weights,
                lateral_weights=lateral_weights,
                longitudinal_weights=longitudinal_weights,
                decoder_params=decoder_student                
            )

        log_dir = Path(weights_path).parents[0]
        print(f"\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m[test] log_dir={log_dir}")

        _x_init = []
        for x in x_val:
            if isinstance(x, list):
                _x_init.append(tf.ragged.constant([x[0]]))
            else:
                _x_init.append(np.array([x[0]]))
        model(_x_init)
        model.load_weights(weights_path)

        
        test_data = load_data(test_dir)
        x_test, y_test = preprocessing(test_data, input_shape)
        del test_data
        
        print_shape(mode='test', x=x_test, y=y_test)
        
        print("\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m[test] evaluating model...")
        
        
        maneuvers_lat = ["LLC", "RLC", "TL", "TR", "LK"]
        maneuvers_lat = [p + m for p in ['true_', 'pred_'] for m in maneuvers_lat]

        maneuvers_lon = ["ST", "ACC", "DEC", "KS"]
        maneuvers_lon = [p + m for p in ['true_', 'pred_'] for m in maneuvers_lon]

        columns = maneuvers_lat + maneuvers_lon
         
        for name, x_true, y_true in zip(["train", "val", "test", "unlabeled"],
                                        [x_labeled, x_val, x_test, x_unlabeled],
                                        [y_labeled, y_val, y_test, None]):
            
            y_pred_file = os.path.join(log_dir, f"{name}_predict.csv")
            
            if y_true is not None:
                eval_logs = model.evaluate(
                        x = x_true,
                        y = y_true,
                        batch_size=1,
                        workers=4,
                        verbose=0,
                        use_multiprocessing=True,
                        return_dict=True,
                    )
                
                eval_file = os.path.join(log_dir, f"{name}_eval.json")
                with open(eval_file, 'w') as f:
                    json.dump(eval_logs, f)
                
                print(f"\t {name}:")
                for metric, value in eval_logs.items():
                    print("\t\t {}={:.3f}".format(metric, value))
            
                lat_pred, lon_pred = model.predict(x_true)
                data_conct = np.concatenate([y_true[0], lat_pred, y_true[1], lon_pred], axis=1)
                pd.DataFrame(data_conct, columns=columns).to_csv(y_pred_file)
            else:
                lat_pred, lon_pred = model.predict(x_true)
                data_conct = np.concatenate([lat_pred, lon_pred], axis=1)
                pd.DataFrame(data_conct, columns=["LLC", "RLC", "TL", "TR", "LK", 
                                                  "ST", "ACC", "DEC", "KS"]).to_csv(y_pred_file)
                



        print("\033[94m[Noisy Student][Graph Maneuver Prediction][main]\033[0m DONE!")