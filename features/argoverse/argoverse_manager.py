import os
import sys
from pathlib import Path

# sys.path.append(os.path.abspath('../../.'))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from typing import Dict, List, Optional, Tuple, NoReturn

from joblib import Parallel, delayed
import shutil
import tempfile
import time
import json
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx

#from argoverse
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

#from utils folder
from utils.interaction_graph import InteractiveGraph, Vehicle
import utils.transform as trf 

#from current folder
from features.argoverse.motion_features import MotionFeatures
from features.argoverse.map_features import MapFeatures
from features.argoverse.maneuver_features import ManeuverFeatures


class ArgoverseManager(object):

    def __init__(self, 
                 root:str, 
                 maneuver_path:str, 
                 filter:str="none" 
    ):
        """
            create the dataset manager object

            - params:
                + root : str : root path of the data
                + maneuver_path : str : path to the maneuver labels
                + filter: str : type of filter used by the Fuzzy Inference System
                                        [none, ekf, savgol]

        """

        self.root_path = root

        self.filter=filter
        self.manager = ArgoverseForecastingLoader(self.root_path)
        
        self.motion_ext   = MotionFeatures(filter=self.filter)
        self.map_ext      = MapFeatures()
        self.maneuver_ext = ManeuverFeatures(labels_path=maneuver_path)
        self.maneuvers_path = maneuver_path

    def _build_spatial_graph_from_sequence(
            self, 
            seq:str,
            obs_len:int,
            interaction_field:int,
            max_neighbors:int
    )->Tuple[\
            List[str],#sequences
            List[np.ndarray],#input_target
            List[np.ndarray],#input_graph
            List[np.ndarray],#output_target
            InteractiveGraph
    ]:
        
        sequence_df = self.manager.get(seq).seq_df
        
        unique_ids = sequence_df['TRACK_ID'].unique()
        
        _graph = InteractiveGraph(
                    interaction_field=interaction_field,
                    name=seq.name
                )
        
        ego_pose = sequence_df[sequence_df['OBJECT_TYPE']=='AV'][['X', 'Y']].values[0]
        
        for idx, idn in enumerate(unique_ids):
            #data for a specific agent
            _seq_v = sequence_df[sequence_df['TRACK_ID']==idn]
            
            #filter: track is not long enough
            if _seq_v.shape[0] < obs_len*10:
                continue            
            
            #create vehicle
            _v = Vehicle(name=idn,id=idx)    
            
            #(x,y, theta, vx, vy)            
            traj, _ = self.motion_ext.get_trajectory(seq=_seq_v, obs_len=5)
            traj[:, 0:2] = trf.translate(seq=traj[:, 0:2], T=-ego_pose)          
                
            _v.historical_trajectory = traj[:obs_len*10] 
            _v.future_trajectory = traj[obs_len*10:]

            #get agent type
            if (_seq_v['OBJECT_TYPE'] == 'AGENT').all():
                _v.type = Vehicle.Category.TARGET_VEHICLE
                
                if len(_v.future_trajectory)>0:
                    _v.future_trajectory[:, :2] =\
                        trf.translate(seq=_v.future_trajectory[:, :2], 
                                        T=-_v.historical_trajectory[-1, :2])
                

                #lane deviation: road geometric features    
                _v.lane_deviation, _v.lane = self.map_ext.get_lane_deviation(
                    seq = _seq_v,
                    obs_len=obs_len,
                    padding_lane_geo=True,
                    lane_length=len(traj)
                )
                
                _v.lane = trf.translate(seq=_v.lane, T=-ego_pose)

                #lat-lon maneuver intention
                (lat_intention, lat_probs), (lon_intention, lon_probs)=\
                    self.maneuver_ext.get_intentions_from_labels(
                        seq=seq.name
                    )
                            
                _v.lat_maneuver_intention = lat_intention
                _v.lat_maneuver_probs = lat_probs

                _v.lon_maneuver_intention = lon_intention
                _v.lon_maneuver_probs = lon_probs
            
            elif (_seq_v['OBJECT_TYPE'] == 'AV').all():
                _v.type = Vehicle.Category.EGO_VEHICLE
                
            else:
                _v.type = Vehicle.Category.GENERIC_VEHICLE
            
            
            #add to graph
            _graph.add_vehicle(_v)
       
        _graph.build_weighted_edges()  
        _graph = _graph.get_graph_from_interaction_field(max_neighbors=max_neighbors+1)
        _graph.update_keys()
        #edges
        edges, nodes_id, adj_matrix =\
             _graph.get_edges_using_interaction_field()      
    
        #features 
        quant_nodes = len(nodes_id)
        
        _sequences = []
        _input_target = []
        _input_graph = []
        _output_target = []
        
        _sequences.append(_graph.name)
        
        _input_graph.append(_graph.target_vehicle)
        _input_graph.append(edges)
        _input_graph.append(-1*np.ones((quant_nodes, quant_nodes)))
                
        nodes_hist_traj = np.zeros((quant_nodes, obs_len*10, 5))
        
        for node_id in _graph.nodes:
            node = _graph.nodes[node_id].get("content")
            if node.type == Vehicle.Category.TARGET_VEHICLE:
                _input_target.append((node.historical_trajectory,
                                      node.lane,
                                      node.lane_deviation))
                
                _output_target.append(node.future_trajectory)
                
            nodes_hist_traj[node_id] = node.historical_trajectory
            
            _input_graph[2][node_id]=\
                adj_matrix[np.where(nodes_id==node_id)[0]]
        
        _input_graph.append(nodes_hist_traj)     
    
              
        return (_sequences, 
                _input_target,
                _input_graph,
                _output_target,
                _graph)
        
    def _build_spatial_graphs(
            self,
            obs_len:int,
            batch_size:int,
            interaction_field:int,
            max_neighbors:int
    )->Tuple[List[str],#sequences
             List[np.ndarray],#input_target
             List[np.ndarray],#input_graph
             List[np.ndarray],#output_target
             List[InteractiveGraph],#graphs
    ]:
        
        """
        build spatial graphs for a sequence of scenarios

        Args:
            mode (str): [sample/train/val/test]
            obs_len (int): [size of the observation]
            batch_size (int): [size of the batch for parallel processing]
            interaction_field (int): radius of the interaction field
            max_neighbors (int): max number of neighbors
        Returns:
            List[InteractiveGraph] : list of graphs
            List[Dict] : list of features
        """       
        
        ## aux function for parallel processing
        def _aux(
            start_idx:int
        ) -> Tuple[\
                List[str],#sequences
                List[np.ndarray],#input_target
                List[np.ndarray],#input_graph
                List[np.ndarray],#output_target
                List[InteractiveGraph]#graphs
        ]:
        
            _graphs=[]
            _in_sequences=[]
            _in_target=[]
            _in_graph=[]
            _out_target=[]

            for seq in sequences[start_idx:start_idx+batch_size]:
                if seq.suffix != ".csv":
                    continue
                
                results = self._build_spatial_graph_from_sequence(
                            seq=seq,
                            obs_len=obs_len,
                            interaction_field=interaction_field,
                            max_neighbors=max_neighbors
                        )
                
                _in_sequences.append(results[0])    
                _in_target.append(results[1])
                _in_graph.append(results[2])
                _out_target.append(results[3])
                _graphs.append(results[4])
            
            return (_in_sequences,
                     _in_target,
                     _in_graph,
                     _out_target,
                     _graphs)
        #######
        
        sequences = self.manager.seq_list
        batch_size = max(1, min(batch_size, len(sequences)//5))
        
                
        #build graphs using parallel computation
        result = Parallel(n_jobs=-2)(delayed(_aux)
            (
                start_idx=i                   
            ) for i in tqdm(range(0, len(sequences), batch_size)))
        
        
        r_seq  = np.concatenate([p[0] for p in result])
        
       
        r_in_target = [(item[0], item[1], item[2]) 
                       for patch in result 
                       for batch in patch[1]
                       for item in batch]
       
                
        r_in_graph  = [(item[0], item[1], item[2], item[3]) 
                       for batch in result 
                       for item in batch[2]]
        
        r_out_target = np.concatenate([np.asarray(p[3], dtype=object) 
                                 for p in result]).squeeze()
        
       
        r_graph = np.concatenate([np.asarray(p[4], dtype=object) 
                                 for p in result])
        
        return (r_seq,
                r_in_target,
                r_in_graph,
                r_out_target,
                r_graph)
        

    def process(
            self, 
            save_dir:str,
            mode:str,
            obs_len:int=2,
            batch_size:int=100,
            interaction_field:int=100,
            max_neighbors:int=20
    ) -> NoReturn:
        
        """
        extract features from the Argoverse dataset

        Args:
            save_dir (str): [folder to save the features]
            mode (str): [sample/train/val/test]
            obs_len (int, optional): [size of the observation in seconds]. Defaults to 2.
            batch_size (int, optional): [size of the batch for extracting the features]. Defaults to 100.

        Returns:
            NoReturn
        """

        if mode == 'test':
            obs_len = 2 #2 seconds
        
        print("[Argoverse][process] building the spatial graphs....")
        
        results = self._build_spatial_graphs(
                        batch_size=batch_size,
                        obs_len=obs_len,
                        interaction_field=interaction_field,
                        max_neighbors=max_neighbors
                )
        
        print("[Argoverse][process] saving spatial graphs....")
        
        paths = ['sequences.npy',
                 'in_target.ser', 
                 'in_graph.ser', 
                 'out_target.npy',
                 'graphs.ser']
        
        base_dir = f"{self.filter}_{mode}" if self.filter != "none" else\
                   mode
        base_dir = os.path.join(save_dir, base_dir)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir) 

        for file, data in zip(paths, results):
            save_path = os.path.join(base_dir, file)
            
            if file.endswith(".npy"):
                np.save(save_path, data)
            else:    
                with open(save_path, 'wb') as f:
                    pkl.dump(data, f)

        new_maneuvers_path = os.path.join(base_dir, "maneuvers.csv")
        shutil.copy(self.maneuvers_path, new_maneuvers_path)
        
        print("[Argoverse][process] done!")

        