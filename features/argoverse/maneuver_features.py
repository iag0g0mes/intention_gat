import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path

# sys.path.append(os.path.abspath('../../.'))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

from features.argoverse.motion_features import MotionFeatures



class ManeuverFeatures(object):
    
    def __init__(self, labels_path:str=""):
       
        self.maneuver = {}
        self.maneuver['lateral'] = {}
        self.maneuver['longitudinal'] = {}
        
        lat_names = ['LLC', 'RLC', 'TL', 'TR', 'LK']
        self.maneuver['lateral']={n:v for v,n in enumerate(lat_names)}
        
        lon_names = ['ST', 'ACC', 'DEC', 'KS']
        self.maneuver['longitudinal']={n:v for v,n in enumerate(lon_names)}
        
        self.labels = None
        
        self.labels_path = labels_path
        if self.labels_path:
            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(("[ManeuverFeatures][__init__]"
                                     f"labels_path not found:{self.labels_path}"))
            
            self.labels = pd.read_csv(self.labels_path)
        
    def set_labels_file(self, file):
        if not os.path.exists(file):
            raise FileNotFoundError(("[ManeuverFeatures][set_labels_file]"
                                     f"file not found:{file}"))
            
        self.labels_path=file
        self.labels = pd.read_csv(file)
        
       
    def get_intentions_from_labels(
            self, 
            seq:str
    )->Tuple[Tuple[int, List[float]],Tuple[int, List[float]]]:
        """
        
        get the maneuver intention and probabilities of a sequence file
            from the labels

        Args:
            seq (string): name of the sequence file (.csv)
            
        Returns:
            List (int) : lat
            List ([float]): probabilities for each maneuver
        """
        
        lat = []
        lat_probs = []
        lon = []
        lon_probs = []
        
        if self.labels is not None:
            if seq not in self.labels['FILE'].values:
                raise ValueError(("[ManeuverFeatures][get_intentions_from_labels]"
                                  f"seq not found in labels. (seq:{seq})"))
                
            lat = self.labels[self.labels['FILE']==seq]['LAT'].values.squeeze()
            lat_probs = self.labels[self.labels['FILE'] == seq]\
                                    [['LLC', 'RLC', 'TL', 'TR', 'LK']].values.squeeze()
        
            lon = self.labels[self.labels['FILE']==seq]['LON'].values.squeeze()
            lon_probs = self.labels[self.labels['FILE'] == seq]\
                                    [['ST', 'ACC', 'DEC', 'KS']].values.squeeze()
        
            lat = str(lat)
            lon = str(lon)
            
            if (not lat is np.nan) and (lat in self.maneuver['lateral']):
                lat = str(lat)
                lat = self.maneuver['lateral'][lat]
            else:
                lat = np.nan
                lat_probs=np.zeros(5)

            if (not lon is np.nan)  and (lon in self.maneuver['longitudinal']):
                lon = str(lon)
                lon = self.maneuver['longitudinal'][lon]
            else:
                lon = np.nan
                lon_probs = np.zeros(4)
            
        return (lat, lat_probs), (lon, lon_probs)
 
    
    
    def get_longitudinal_maneuver(
        self,
        traj:np.ndarray
    )->Tuple[np.ndarray, np.ndarray]:
        """
        estimate longitudinal maneuver for a given trajectory
        
        maneuvers:
            - STOP
            - ACCELERATE
            - DECELERATE
            - KEEP_SPEED
        trajectory:
            - (x, y, theta, vx, vy)

        Args:
            traj (np.ndarray): [trajectory (n, 5)] -> (x, y, theta, vx, vy)

        Returns:
            np.ndarray: [description]
        """
        vx = traj[:, 3]
        vy = traj[:, 4]
        v = np.sqrt(np.power(vx, 2) + np.power(vy, 2))
        
        #stop, acc, dec, keep_speed
        maneuver = np.zeros(4)
        
        #stop
        size = len(v)
        slice = size//5

        mu_v_o = np.mean(v[:slice])
        mu_v_f = np.mean(v[-slice:])

        if (mu_v_f<=0.5): #stop
            maneuver[0] = 1.
        elif abs(mu_v_o - mu_v_f)<=2.0:#keep_speed
                maneuver[3]=1.
        elif mu_v_o < mu_v_f:#acc
                maneuver[1] = 1.
        else:#dec
                maneuver[2] = 1.
            
        return np.argmax(maneuver), maneuver

