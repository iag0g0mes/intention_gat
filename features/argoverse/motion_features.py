import os 
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

from utils.filters import SavitzkyGolay, EKF


class MotionFeatures(object):
    
    def __init__(self, filter:str="none"):

        if filter == "savgol":
            self.filter = SavitzkyGolay(window_length=5, poly=3)
        elif filter == "ekf":
            self.filter = EKF()
        else:
            self.filter = None
        
    
    def get_trajectory(
            self, 
            seq:pd.DataFrame,
            obs_len:int
    )->Tuple[np.ndarray,np.ndarray]:
        """[summary]

        Args:
            seq (pd.DataFrame): [DataFrame with data]
            obs_len (int) : [observation size]
        Returns:
            historical_traj (np.ndarray): [historical trajectory with position (x,y, theta, vel)]
            future_traj (np.ndarray): [future trajectory with position (x,y,theta, vel)]
        """
        
        traj = seq[['X', 'Y']].values
        
        if self.filter is not None:
            traj = self.filter.process(traj)
        
        vel, vel_x, vel_y = self._get_vel(traj=traj)
        theta = self._get_theta(traj=traj)
          
        traj = np.concatenate((traj, theta, vel_x, vel_y), axis=1)
               
        #10Hz
        hist = traj[0:obs_len*10, :]
        futt = traj[obs_len*10:,  :]
        
        return (hist, futt)
    
    def _get_vel(
            self, 
            traj:np.ndarray,
            padding:bool=True,
    )->Tuple[List[float],List[float],List[float]]:
        """[summary]

        Args:
            traj (np.ndarray): [array with position (x,y)]
            padding (bool): [if the result should be filled util 
                                its size is equal to the input traj]

        Returns:
            vel (List[float]) : [velocity]
            vel_x (List[float]) : [velocity in x-axis]
            vel_y (List[float]) : [velocity in y-axis]
        """
        x = traj[:, 0]
        y = traj[:, 1]
        
        vel_x = np.ediff1d(x)/0.1
        vel_y = np.ediff1d(y)/0.1
        
        vel = np.sqrt(np.power(vel_x,2) + np.power(vel_y,2))
        
        #padding style: reapt the last element of the array
        if padding:
            vel_x = np.pad(
                        vel_x, 
                        constant_values=vel_x[-1],
                        pad_width=(0,len(x)-len(vel))
                    )
            vel_y = np.pad(
                        vel_y, 
                        constant_values=vel_y[-1],
                        pad_width=(0,len(x)-len(vel))
                    )
            vel = np.pad(
                        vel, 
                        constant_values=vel[-1],
                        pad_width=(0,len(x)-len(vel))
                    )
            
        vel = np.expand_dims(vel, axis=1)
        vel_x = np.expand_dims(vel_x, axis=1)
        vel_y = np.expand_dims(vel_y, axis=1)
        
        return (vel, vel_x, vel_y)
        
    def _get_theta(
        self,
        traj:np.ndarray,
        padding:bool=True        
    ) -> List[float]:
        """
        
        estimates the orientation between two points using arctan2

        Args:
            traj (np.ndarray): [array with position (x,y)]
            padding (bool): [if the result should be filled util 
                                its size is equal to the input traj]


        Returns:
            List[float]: [orientation]
        """
    
        x = traj[:, 0]
        y = traj[:, 1]
        
        xx = np.ediff1d(x)
        yy = np.ediff1d(y)
        
        theta = np.arctan2(yy, xx)
        
        #padding style: reapt the last element of the array
        if padding:
            theta = np.pad(
                        theta, 
                        constant_values=theta[-1],
                        pad_width=(0,len(x)-len(theta))
                    )
            
        theta = np.expand_dims(theta, axis=1)
            
        return theta
        
    