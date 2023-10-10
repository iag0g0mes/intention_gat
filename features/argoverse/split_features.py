from cgi import test
import os
import shutil
import sys
import argparse
import configparser
from typing import List, Tuple, Any
from pathlib import Path
from ast import literal_eval as make_tuple
import numpy as np
import pandas as pd
import pickle as pkl
from typing import Any, Dict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.preprocessing import load_data, preprocessing, print_shape
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

def save_features(
    root:str,
    data
):

    if os.path.exists(root):
        shutil.rmtree(root)
    os.mkdir(root)

    paths = ['sequences.npy',
             'in_target.ser', 
             'in_graph.ser', 
             'out_target.npy',
             'graphs.ser',
             'maneuvers.csv']
        
    for file, data in zip(paths, data):
        save_path = os.path.join(root, file)
        if file.endswith(".npy"):
            np.save(save_path, data)
        elif file.endswith('.csv'):
            data.to_csv(save_path)
        else:    
            with open(save_path, 'wb') as f:
                pkl.dump(data, f)

def split_dataset_into_folds(
    root:str,
    data:Dict,
    n_folds:int,
    mode:str,
    stratify:bool=False,
    test_size:float=0.2,
):
    sequences = data['sequences']
    in_graph = data['in_graph']
    in_target = data['in_target']
    out_target = data['out_target']
    graphs = data['graphs']
    maneuvers = data['maneuvers']
    idx_files = [np.where(maneuvers["FILE"]==s[0])[0][0] for s in sequences]
    maneuvers = maneuvers.iloc[idx_files]

    lat_maneuvers = maneuvers['LAT'].values
    if stratify:
        print("[Split Features] stratified split")
        lat_idx = {k:np.where(lat_maneuvers==k)[0] 
                    for k in np.unique(lat_maneuvers)}
        test_idx = []
        if (test_size > 0) and (n_folds > 1) and (mode=="val"):
            test_idx = [
                np.random.choice(v, int(len(v)*test_size), replace=False)
                for _ , v in lat_idx.items()
            ]
            test_idx = np.concatenate(test_idx)
            
            print(f"Saving test set... size: {len(test_idx)}")
            c_test = np.unique(lat_maneuvers[test_idx], return_counts=True)
            for k, v in zip(c_test[0], c_test[1]):
                print(f"\t({k}:{v})->{round(float(v)/len(test_idx), 3)*100}%")

            n_folds = n_folds - 1

            folder_path = os.path.join(root, "test")
            _data = (
                sequences[test_idx],
                [in_target[i] for i in test_idx],
                [in_graph[i] for i in test_idx],
                out_target[test_idx],
                graphs[test_idx],
                maneuvers.iloc[test_idx]
            )

            save_features(folder_path, _data)
        
        fold_size = 1./max(1, n_folds)

        remaining_idx = np.arange(0, len(lat_maneuvers))
        remaining_idx = remaining_idx[np.logical_not(
                                        np.isin(
                                            remaining_idx,
                                            test_idx
                                        )
                                    )]
        count_labels = np.unique(lat_maneuvers[remaining_idx], return_counts=True)
        count_labels = {k:int(v*fold_size) 
                        for k, v in zip(count_labels[0], count_labels[1])}

        used_idx = test_idx
        for nf in range(0, n_folds):
            #update idx
            fold_idx = np.arange(0, len(lat_maneuvers))
            fold_idx = fold_idx[np.logical_not(np.isin(fold_idx, used_idx))]
            lat_idx = {k:np.where(lat_maneuvers[fold_idx]==k)[0] 
                        for k in np.unique(lat_maneuvers)}

            data_idx = [
                np.random.choice(v, count_labels[k], replace=False)
                for k , v in lat_idx.items()
            ]
            data_idx = np.concatenate(data_idx)
            data_idx = fold_idx[data_idx]

            used_idx = np.concatenate((used_idx, data_idx))

            print(f"Saving fold {nf}... size: {len(data_idx)}")
            c_test = np.unique(lat_maneuvers[data_idx], return_counts=True)
            for k, v in zip(c_test[0], c_test[1]):
                print(f"\t({k}:{v})->{round(float(v)/len(data_idx), 3)*100}%")

            folder_path = os.path.join(root, f"fold_{nf}")
            _data = (
                sequences[data_idx],
                [in_target[i] for i in data_idx],
                [in_graph[i] for i in data_idx],
                out_target[data_idx],
                graphs[data_idx],
                maneuvers.iloc[data_idx]
            )

            save_features(folder_path, _data)

    else:
        print("[Split Features] random split")
        indexes = np.arange(0, len(sequences))
        np.random.shuffle(indexes)
        np.random.shuffle(indexes)
        start_idx = 0

        if (test_size > 0) and (n_folds > 1) and (mode=="val"):
            start_idx = int(len(indexes)*test_size)
            test_idx = indexes[:start_idx]

            print(f"Saving test set... size: {len(test_idx)}")
            c_test = np.unique(lat_maneuvers[test_idx], return_counts=True)
            for k, v in zip(c_test[0], c_test[1]):
                print(f"\t({k}:{v})->{round(float(v)/len(test_idx), 3)*100}%")

            n_folds = n_folds - 1

            folder_path = os.path.join(root, "test")
            _data = (
                sequences[test_idx],
                [in_target[i] for i in test_idx],
                [in_graph[i] for i in test_idx],
                out_target[test_idx],
                graphs[test_idx],
                maneuvers.iloc[test_idx]
            )

            save_features(folder_path, _data)


        batch_size = len(indexes[start_idx:])//n_folds
        
        for nf in range(0, n_folds):
            data_idx = indexes[nf*batch_size:(nf+1)*batch_size]

            print(f"Saving fold {nf}... size: {len(data_idx)}")
            c_test = np.unique(lat_maneuvers[data_idx], return_counts=True)
            for k, v in zip(c_test[0], c_test[1]):
                print(f"\t({k}:{v})->{round(float(v)/len(data_idx), 3)*100}%")

            folder_path = os.path.join(root, f"fold_{nf}")
            _data = (
                sequences[data_idx],
                [in_target[i] for i in data_idx],
                [in_graph[i] for i in data_idx],
                out_target[data_idx],
                graphs[data_idx],
                maneuvers.iloc[data_idx]
            )

            save_features(folder_path, _data)

    print("Moving source data to source folder")
    files = ['sequences.npy', 'in_graph.ser', 
             'in_target.ser', 'out_target.npy', 
             'graphs.ser', 'maneuvers.csv']
    folder_path = os.path.join(root, 'source')
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    for file in files:
        source_path = os.path.join(root, file)
        new_path = os.path.join(folder_path, file)
        shutil.move(source_path, new_path)
    c_data = np.unique(lat_maneuvers, return_counts=True)
    for k, v in zip(c_data[0], c_data[1]):
        print(f"\t({k}:{v})->{round(float(v)/len(lat_maneuvers), 3)*100}%")

if __name__ == "__main__":
    """Split Features into K-Folds"""

    print("[Split Features] running.....")

    args = parse_arguments()

    if not os.path.exists(args.cfg):
        raise FileNotFoundError(("config file not found!"
                                 f"(cfg:{args.cfg})"))

    config = configparser.ConfigParser()
    config.read(args.cfg)

    for k, v in config['DIRS'].items():
        if not os.path.exists(config['DIRS'][k]):
            raise FileNotFoundError((f"{k} not found! ({k}:{v})"))

    root = os.path.join(config["DIRS"]["data_dir"], config["PARAMS"]["mode"])
    
    if not os.path.exists(root):
        raise FileNotFoundError(f"features directory not found! ({root})")

    data = load_data(config["DIRS"]["data_dir"], mode=config["PARAMS"]["mode"])#, size=100)

    # input_shape={}
    # input_shape['trajectory'] =\
    #     make_tuple(config["INPUT"]["trajectory"])
    # input_shape['lane_geometry'] =\
    #     make_tuple(config["INPUT"]["lane_geometry"])
    # input_shape['lane_deviation'] =\
    #     make_tuple(config["INPUT"]["lane_deviation"])

    # X, Y = preprocessing(data, in_shape=input_shape)
    # del data

    # print_shape(mode=config["PARAMS"]["mode"], x=X, y=Y)

    n_folds=config["PARAMS"].getint("kfold")
    test_size=config["PARAMS"].getfloat("test_size")
    stratify=config["PARAMS"].getboolean("stratify")

    split_dataset_into_folds(
        root=root,
        data=data,
        mode=config["PARAMS"]["mode"],
        n_folds=n_folds,
        test_size=test_size,
        stratify=stratify
    )
    