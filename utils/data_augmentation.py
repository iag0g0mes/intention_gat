import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from typing import List, Tuple, Any, Dict, NoReturn
from tqdm import tqdm
import matplotlib.pyplot as plt

# sys.path.append(os.path.abspath('../../.'))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import utils.transform as trf

from features.argoverse.map_features import MapFeatures
from features.argoverse.motion_features import MotionFeatures
from features.argoverse.maneuver_features import ManeuverFeatures

def __intermittent_noise(x:np.ndarray)-> np.ndarray:
    rng = np.random.default_rng()

    _x = np.copy(x)
    noise = rng.normal(loc=0, 
                       scale=0.5, 
                       size=(x.shape[0], x.shape[1]))

    idx_zero = rng.choice(range(0, x.shape[0]), 
                          size=int(x.shape[0]*0.8), 
                          replace=False)
    noise[idx_zero]=0
    
    _x = _x + noise
    
    return _x

def __spikes_noise(x:np.ndarray)->np.ndarray:
    rng = np.random.default_rng()

    _x = np.copy(x)
    try:
        spike_size = rng.integers(1,int(len(_x)*0.15))

        noise = rng.normal(loc=0,
                        scale=0.5,
                        size=(spike_size, x.shape[1])
                        )

        idx = rng.integers(0, _x.shape[0]-spike_size)
        
        _x[idx:idx+spike_size] += noise
    except Exception:
        print(np.shape(_x))
        assert False
    return _x
    
def __continuous_noise(x:np.ndarray,
                       quant:int=1
)->np.ndarray:
    rng = np.random.default_rng()

    _x = np.copy(x)
    _x = _x + rng.normal(loc=0, 
                         scale=0.3, 
                         size=(x.shape[0], x.shape[1]))
    
    return _x

def __rotate(x:np.ndarray, max_rot:float=45.)-> np.ndarray:
    rng = np.random.default_rng()
    rot = 2*np.pi*rng.random()/(360./max_rot)
    _x = np.copy(x)
    T = [-x[0,0], -x[0, 1]]

    _x[:, 0:2] = trf.translate(_x[:, 0:2], T=T)
    _x[:, 0:2] = trf.rotate(_x[:, 0:2], rot=rot)

    T=[_x[0, 0]-T[0], _x[0, 1]-T[1]]
    _x[:, 0:2] = trf.translate(_x[:, 0:2], T=T)   
    
    return _x

# def __translate(x:np.ndarray,
#           ylat:np.ndarray, 
#           ylon:np.ndarray, 
#           quant:int
# )->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
#     #mu=0, var=1
#     T = 30*np.random.randn(quant, 2) - 10
    
#     t_x = []
#     t_lat = []
#     t_lon = []
    
#     for t in T:
#         idx = np.random.randint(0, x.shape[0])
#         _x = np.copy(x[idx])
        
#         _nx = trf.translate(_x[:, 0:2], T=t)
#         _nx = np.concatenate((_nx, _x[:,2:]), axis=1)
             
#         t_x.append(_nx)
#         t_lat.append(ylat[idx])
#         t_lon.append(ylon[idx])
    
#     t_x = np.asarray(t_x)
#     t_lat = np.asarray(t_lat)
#     t_lon = np.asarray(t_lon)    
    
#     return (t_x, t_lat, t_lon)

def transform(
    X:Tuple[np.ndarray, np.ndarray, np.ndarray, List, List],
    opt:str
)->Tuple[np.ndarray, np.ndarray, np.ndarray, List, List]:

    operations = ['conti_noise', 'inter_noise', 'spike_noise', 'rotate']
    f_functions = [__continuous_noise, __intermittent_noise, __spikes_noise, __rotate]

    if opt not in operations:
        raise ValueError((f"Data augmentation operation not valid: {opt}!"
                          f"\n valid options: {operations}"))
    
    f_metric = {k:v for k, v in zip(operations, f_functions)}
    
    x_traj = X[0]
    x_geo = X[1]
    x_dev = X[2]
    x_edges = X[3]
    x_nodes = X[4]

    if opt=='rotate':
        x_traj = f_metric[opt](x=x_traj, max_rot=30.)
        x_geo = f_metric[opt](x=x_geo, max_rot=30.)
        x_nodes = [f_metric[opt](x=t, max_rot=15.) for t in x_nodes]
    else:
        print("transform", opt, end=" ")
        print(np.shape(x_traj), end=" ")
        print(np.shape(x_geo), end=" ")
        print(np.shape(x_dev), end=" ")
        _s = [np.shape(t) if t is not None else None for t in x_nodes]
        print(_s)

        x_traj = f_metric[opt](x=x_traj)
        x_geo = f_metric[opt](x=x_geo)
        x_dev = f_metric[opt](x=x_dev)
        x_nodes = [f_metric[opt](x=t) for t in x_nodes]

    return (x_traj, x_geo, x_dev, x_edges, x_nodes)


def __balanced_indexes(
    y_lat:np.ndarray,
    y_lon:np.ndarray,
    samples_per_class:int,
    random_selection:bool=True
) -> np.ndarray:
    rng = np.random.default_rng()
    
    arg_y_lat = np.argmax(y_lat, axis=1)
    classes_lat =\
        { k: np.where(arg_y_lat==k)[0]
            for k, n in zip(*np.unique(arg_y_lat, 
                                        return_counts=True
                                        )
                            )
        }
    if random_selection:
        for c_lat, idx_lat in classes_lat.items():
            arg_y_lon = np.argmax(y_lon[idx_lat], axis=1)
            unq_y_lon = np.unique(arg_y_lon, return_counts=True)
            p_class = np.zeros(len(idx_lat))

            for c_lon, len_lon in zip(*unq_y_lon):
                idx_lon = np.where(arg_y_lon==c_lon)[0]
                p_class[idx_lon] =\
                    1.0/(len(unq_y_lon[0])*len_lon)
            p_class = p_class/np.sum(p_class)

            idx_sequence = rng.choice(idx_lat,
                                  size=samples_per_class,
                                  replace=True,
                                  p=p_class)
            classes_lat[c_lat]  = idx_sequence
    else:
        for c_lat, idx_lat in classes_lat.items():
            arg_y_lon = np.argmax(y_lon[idx_lat], axis=1)
            unq_y_lon = np.unique(arg_y_lon, return_counts=True)

            quant_per_lon = samples_per_class//len(unq_y_lon[0])

            idx_selected = []
            
            for c_lon, len_lon in zip(*unq_y_lon):
                idx_lon = np.where(arg_y_lon==c_lon)[0]

                if quant_per_lon <= len(idx_lon):
                    idx_random = rng.choice(idx_lon, size=quant_per_lon, replace=False)
                    idx_selected+=list(idx_random)
                else:
                    step = quant_per_lon//len(idx_lon)
                    idx_selected += list(idx_lon)*step

            rest = samples_per_class - len(idx_selected)
            if rest> 0:
                idx_lon = rng.choice(np.arange(0, len(arg_y_lon)), size=int(rest))    
                idx_selected += list(idx_lon)

            classes_lat[c_lat] = idx_lat[idx_selected]

    return classes_lat

def data_augmentation(
    X:Tuple,
    Y:Tuple[np.ndarray, np.ndarray],
    operations:List[str], 
    opt_probs:List[float],
    samples_per_class:int=10000,
)->Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, List, List],
    Tuple[np.ndarray, np.ndarray]
]:

    possible_opts = ['translate', 
                     'conti_noise', 'inter_noise', 
                     'spike_noise', 'rotate']
        
    if not np.all([m in possible_opts for m in operations]):
        raise ValueError((f'Unrecognized value for `metrics`. Received: {operations}'
                           'Expected values are ["conti_noise", "translate", "inter_noise",'
                           ' "rotate", "spike_noise"]'))

    
    #indexes
    y_lat, y_lon = Y[0], Y[1]

    classes_lat = __balanced_indexes(
                        y_lat=y_lat, 
                        y_lon=y_lon,
                        samples_per_class=samples_per_class
                    )
    #shuffle
    rng = np.random.default_rng()

    #data augmentation
    X_new = []
    Y_lat_new = []
    Y_lon_new = []

    for c_i, (c, idx_lat) in enumerate(classes_lat.items()):
    
        for i in idx_lat:
            _sample = [X[j][i] for j in range(0, len(X))]
            for opt, prob in zip(operations, opt_probs):
                if rng.uniform(low=0, high=1) <= prob:
                    _sample = transform(_sample, opt)

            X_new.append(_sample)
            Y_lat_new.append(y_lat[i])
            Y_lon_new.append(y_lon[i])
            
    #format
    Y_lat_new = np.asarray(Y_lat_new)
    Y_lon_new = np.asarray(Y_lon_new)
    
    traj = np.vstack([[X_new[i][0]] for i in range(len(X_new))])   
    geo = np.vstack([[X_new[i][1]] for i in range(len(X_new))])
    dev = np.vstack([[X_new[i][2]] for i in range(len(X_new))])
    edges = [X_new[i][3] for i in range(len(X_new))]
    nodes = [X_new[i][4] for i in range(len(X_new))]

    X_new = [traj, geo, dev, edges, nodes]
    return X_new, [Y_lat_new, Y_lon_new]
