import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import pickle as pkl
import tensorflow.keras.backend as K
from typing import Dict, Tuple, List


def onehot_maneuver(maneuver)->np.ndarray:
    y_max= np.argmax(maneuver, axis=1)
    y_max=np.expand_dims(y_max, axis=1)
    
    onehot_intention = OneHotEncoder()
    onehot_intention.fit(np.expand_dims(
        range(0, maneuver.shape[1]),
        axis=1
    ))
    intent = onehot_intention.transform(y_max).toarray()

    return intent

def onehot_intention(lat, lon)->np.ndarray:
    #one_hot encoding for maneuver intention
    #lat = LLC, RLC, TL, TR, LK
    #lon = ST, ACC, DEC, KS
    #class_num = lat + lon = 5 + 4
    
    num_lat = lat[0].shape[0]
    num_lon = lon[0].shape[0]
    
    disc_lat = np.argmax(lat, axis=1)
    disc_lon = np.argmax(lon, axis=1)
    
    intent_class = (num_lat - 1)*disc_lat + disc_lon
    intent_class = np.expand_dims(intent_class, axis=1)
    
    onehot_intention = OneHotEncoder()
    onehot_intention.fit(np.expand_dims(
                            range(0, num_lat*num_lon), 
                            axis=1)
                            )
    intent = onehot_intention.transform(intent_class).toarray()
    
    return intent


def tensor_onehot_intention(lat, lon)->np.ndarray:
    #one_hot encoding for maneuver intention
    #lat = LLC, RLC, TL, TR, LK
    #lon = ST, ACC, DEC, KS
    #class_num = lat + lon = 5 + 4
    
    num_lat = lat[0].shape[0]
    num_lon = lon[0].shape[0]
    
    disc_lat = K.argmax(lat, axis=1)
    disc_lon = K.argmax(lon, axis=1)
    
    intent_class = (num_lat - 1)*disc_lat + disc_lon
    # intent_class = K.expand_dims(intent_class, axis=-1)
    
    intent =K.one_hot(
        indices=intent_class,
        num_classes=num_lat*num_lon
    )
    
    return intent

def load_data(
    root:str, 
    mode:str="",
    size:int=-1
) -> Dict:
    #load data
    if mode:
        data_folder = os.path.join(root, mode)
    else:
        data_folder = root

    if not os.path.exists(data_folder):
        raise FileNotFoundError(('Data folder not found!'
                                 f'data_folder: {data_folder}'))
        
    files = ['sequences.npy', 'in_graph.ser', 
             'in_target.ser', 'out_target.npy', 
             'graphs.ser']
    
    data = {}

    for file in files:
        file_path = os.path.join(data_folder, file)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(('File not found!'
                                 f'file: {file}'))
            
        if file.endswith('.npy'):    
            data[file.split('.')[0]] = np.load(file_path, allow_pickle=True)
            
        elif file.endswith('.ser'):
            with open(file_path, 'rb') as f:
                data[file.split('.')[0]] = pkl.load(f)
                
        if size > 0:
            data[file.split('.')[0]] = data[file.split('.')[0]][:size]
    
    #load maneuvers labels
    data_file = os.path.join(data_folder, f"maneuvers.csv")

    data['maneuvers'] = pd.read_csv(data_file)
    
    
    return data 

def preprocessing(
    data:Dict,
    in_shape:Dict, 
    return_sequence:bool=False
)->Tuple[
    Tuple[np.ndarray, #target trajectory
          np.ndarray, #lane geometry
          np.ndarray, #lane deviation
          np.ndarray, #edges
          np.ndarray],#nodes_states
    Tuple[np.ndarray, #future trajectory
          np.ndarray, #lat intention 
          np.ndarray] #lon intention
]:#node states
    
    #input
    traj_hist = []
    lane_geo  = []
    lane_dev  = []
    
    for d in data['in_target']:
        traj_hist.append(d[0])
        lane_geo.append(d[1])
        lane_dev.append(d[2])
        
    traj_hist = np.asarray(traj_hist, dtype=np.float32)
    lane_geo  = np.asarray(lane_geo, dtype=np.float32)
    lane_dev  = np.asarray(lane_dev, dtype=np.float32)
        
    traj_hist =\
        traj_hist[:, :in_shape['trajectory'][0], :in_shape['trajectory'][1]] 
    lane_geo =\
        lane_geo[:, :in_shape['lane_geometry'][0], :in_shape['lane_geometry'][1]] 
    lane_dev =\
        lane_dev[:, :in_shape['lane_deviation'][0], :in_shape['lane_deviation'][1]] 

    edges = [np.asarray(d[1], dtype=np.int32) for d in data['in_graph']]
    nodes_states = [np.asarray(d[3], dtype=np.float32) for d in data['in_graph']]
    nodes_states = [n[:, :in_shape['trajectory'][0], :in_shape['trajectory'][1]] for n in nodes_states]
   
    # assert False
    #outputs
    out_lat  = []
    out_lon  = []
    labels = data['maneuvers']
    seqs = []
    for seq in data['sequences']:
        row = labels[labels['FILE']==seq[0]]
        if len(row)>0:
            row=row.iloc[0]
        lat = row[['LLC', 'RLC', 'TL', 'TR', 'LK']].to_numpy()
        lon = row[['ST', 'ACC', 'DEC', 'KS']].to_numpy()
        out_lat.append(lat)
        out_lon.append(lon)
        seqs.append(seq[0])
    
    out_lat = np.squeeze(out_lat)
    out_lon = np.squeeze(out_lon)
    out_lat = np.asarray(out_lat, dtype=np.float32)
    out_lon = np.asarray(out_lon, dtype=np.float32)
    seqs = np.array(seqs)
    
    if not return_sequence:
        return ([traj_hist, lane_geo, lane_dev, edges, nodes_states],
                [out_lat, out_lon])
    else:
        return ([traj_hist, lane_geo, lane_dev, edges, nodes_states],
                [out_lat, out_lon],
                seqs)
                
def print_shape(mode, x, y=None):
    head = "\033[94m[Features Shape][Graph Maneuver Prediction]\033[0m"
    
    x_names = ['historical_trajectory', 
               'lane_geometry', 
               'lane_deviation', 
               'edges', 
               'nodes_state']
    y_names = ['lateral_intention',
               'longitudinal_intention']
    
    map_lat = {n:v for n, v in enumerate(['LLC', 'RLC', 'TL', 'TR', 'LK'])}
    map_lon = {n:v for n, v in enumerate(['ST', 'ACC', 'DEC', 'KS'])}

    print(f'{head}[dataset] {mode}:')
    for idx, value in enumerate(zip(x, x_names)):
        if isinstance(value[0], List):
            shape = len(value[0])
        else:
            shape = np.shape(value[0])
        print(f'\t{value[1]}:{shape}')

    if y is not None:
        for value, name in zip(y, y_names):
            print(f'\t{name}:{np.shape(value)}')
            unique_labels = np.unique(np.argmax(value, axis=1), 
                                      return_counts=True)
            for c, n in zip(*unique_labels):
                cn = map_lat[c] if name.startswith("lateral") else\
                     map_lon[c]

                print(f'\t\t{cn}:[{n}/{round(n/float(len(value))*100, 2)}%]')