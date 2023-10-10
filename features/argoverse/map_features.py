from math import dist
import numpy as np
import pandas as pd

from typing import Tuple, List

from shapely.ops import unary_union
from shapely.geometry import LineString, Point, Polygon

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import remove_overlapping_lane_seq

import matplotlib.pyplot as plt

class MapFeatures(object):
    
    def __init__(self):
                        
        self.avm = ArgoverseMap()
        
    def _get_point_in_polygon_score(
        self,
        lane:np.ndarray,
        traj:np.ndarray,
    )->int:
        """
        count the number of point in trajectory that also lies on lane polygon

        Args:
            lane (np.ndarray): [lane centerline]
            traj (np.ndarray): [trajectory]

        Returns:
            int: [number of points in polygon score]
        """
        
        polygon_lane = Polygon(lane)
        
        point_in_polygon_score = 0
        for xy in traj:
            point_in_polygon_score += polygon_lane.contains(Point(xy))
        
        return point_in_polygon_score
            
        
    def _sort_lanes_based_in_point_in_polygon_score(
        self,
        traj:np.ndarray,
        lanes:List[np.ndarray]
    )->Tuple[List[np.ndarray], List[int]]:
        """
        sort a list of lanes centerline based on the number of points
        of the trajectory that lie within the centerlines

        Font: 
            - sort_lanes_based_on_point_in_polygon_score
            - url: https://github.com/jagjeet-singh/argoverse-forecasting/blob/master/utils/map_features_utils.py

        Args:
            traj (np.ndarray): [trajectory (x,y)]
            lanes (List[np.ndarray]): [lanes centerlines]

        Returns:
            Tuple[List[np.ndarray], List[int]]: [lanes centerlines, scores]
        """

        point_in_polygon_scores = []
        for lane in lanes:
            point_in_polygon_scores.append(
                self._get_point_in_polygon_score(
                    lane=lane,
                    traj=traj
                )
            )
            
        randomized_tiebreaker =\
            np.random.random(len(point_in_polygon_scores))
        
        sorted_point_in_polygon_scores_idx =\
            np.lexsort(
                (randomized_tiebreaker,
                np.array(point_in_polygon_scores))
            )[::-1]
        
        sorted_lanes = [
            lanes[i] for i in sorted_point_in_polygon_scores_idx
        ]
        sorted_scores = [
            point_in_polygon_scores[i] for i in sorted_point_in_polygon_scores_idx
        ]
        
        return (sorted_lanes, sorted_scores)

    def _get_lane_centerline(
        self,
        traj:np.ndarray,
        city_name:str,
        obs_len:int,
        max_search_radius:float=50.0,
    )->Tuple[List[np.ndarray], List[int]]:
        """
        search for lanes centerline in the map that are
        close to the vehicle trajectory
        
        Args:
            traj (np.ndarray): [vehicle trajectory (x,y)]
            city_name (str): [name of the city]
            obs_len (int): [size of observation]
            max_search_radius (float): [max distance for searching]
        Returns:
            List[np.ndarray]: [list of lanes centerlines]
            List[int]: point in polygon scores for the lanes centerlines
        """
        
      
        lanes_centerline =\
            self.avm.get_candidate_centerlines_for_traj(
                xy=traj,
                city_name=city_name,
                viz=False,
                max_search_radius=max_search_radius)
        
        scores = 0
        
        #get the best lane centerline based on 
        #    point in polygon score
        if len(lanes_centerline) > 1:
            lanes_centerline, scores =\
                self._sort_lanes_based_in_point_in_polygon_score(
                    lanes = lanes_centerline,
                    traj  = traj
                )
        else:
            
            scores = [self._get_point_in_polygon_score(
                lane=lanes_centerline[0],
                traj=traj
            )]
                        
        
        return (lanes_centerline, scores)    
        
              
    def get_lane_deviation(
        self,
        seq:pd.DataFrame,
        obs_len:int,
        padding_lane_geo:bool=False,
        lane_length:int=-1
    )->Tuple[np.ndarray, np.ndarray]:
        """
        estimate a pointwise distance vector between the agent trajectory and 
            lane centerline

        Args:
            seq (pd.DataFrame): [agent trajectory]
            obs_len (int): [size of the observation]

        Returns:
            List[float]: [distance between the vehicle trajectory and lane centerline]
            np.ndarray: [lane geometry points (x, y)]
        """
        traj = seq[['X', 'Y']].values
        hist_traj = traj[:obs_len*10]

        if lane_length<=0:
            lane_length=len(hist_traj)
        
        #DataFrame: (TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, CITY_NAME)
        city_name = seq['CITY_NAME'].iloc[0]
        
        #get lane centerline
        lanes_centerline, scores = self._get_lane_centerline(
                                    traj=hist_traj,
                                    city_name=city_name,
                                    obs_len=obs_len                                   
                            )
        
        #get the best lane
        lane = lanes_centerline[0]
        
        #estimate lane deviation
        lane_deviation = [
            np.sqrt(np.power(p - lane,2)).sum(axis=1).min()
            for p in hist_traj
        ]
        lane_deviation = np.asarray(lane_deviation)
        lane_deviation = np.expand_dims(lane_deviation, axis=1)
        
        if padding_lane_geo:
            s_lane = np.shape(lane)[0]
            
            if s_lane >= lane_length:
                lane = lane[:lane_length, :]
            else:
                
                lane_x = np.pad(
                        lane[:,0], 
                        constant_values=lane[-1, 0],
                        pad_width=(0,lane_length - s_lane)
                    )
                lane_y = np.pad(
                        lane[:,1], 
                        constant_values=lane[-1, 1],
                        pad_width=(0,lane_length - s_lane)
                    )
                lane_x=np.expand_dims(lane_x, axis=1)
                lane_y=np.expand_dims(lane_y, axis=1)
                lane = np.concatenate((lane_x, lane_y), axis=1)
        
        return lane_deviation, lane
         
        