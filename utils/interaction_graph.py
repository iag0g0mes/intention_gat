
import numpy as np
from typing import Any, Dict, List, Final, NoReturn, Type
import networkx as nx
from enum import Enum

class Agent(object):
    
    def __init__(self, name:str, id:int):
        """
        Class of an generic traffic participant
        
        Args:
            name (str): [name of the traffic participant]
            id (int): [unique identifier of the traffic participant]
        
        Attributes:
            historical_trajectory (np.ndarray): [array with observed trajectory]
                                                [x,y,theta,vel]
        """
        
        self.name:str = name
        self.id:int   = id
        self.historical_trajectory:np.ndarray = None
        self.future_trajectory:np.ndarray = None

class Vehicle(Agent):

    class Category(Enum):
        TARGET_VEHICLE = 1
        SURROUNDING_VEHICLE= 2
        GENERIC_VEHICLE = 3
        EGO_VEHICLE = 4

    def __init__(self, name:str,  id:int):
        """
        Class of a specific traffic participant
        
        Args:
            name (str): [name of the vehicle]
            id (int): [unique identifier of the vehicle]
        
        Traffic Participant: Vehicle
        
        Types of Vehicles: 
            TARGET_VEHICLE (final int) = 1
            SURROUNDING_VEHICLE (final int) = 2
            GENERIC_VEHICLE (final int) = 3 [unspecified vehicle]
            EGO_VEHICLE (final int) = 4 [autonomous vehicle]
        """
        super().__init__(name=name, id=id)
            
        self.type:int = self.Category.GENERIC_VEHICLE
        
        self.lane_deviation:np.ndarray = None
        self.lane:np.ndarray = None
        
        #['LLC', 'RLC', 'TL', 'TR', 'LK']
        self.lat_maneuver_intention:int = -1
        self.lat_maneuver_probs:np.ndarray = np.zeros(5)

        #['ST', 'ACC', 'DEC', 'KS']
        self.lon_maneuver_intention:int = -1
        self.lon_maneuver_probs:np.ndarray = np.zeros(4)
        
        
    def to_dict(self)->Dict:
        
        vehicle = {}
        vehicle['type'] = self.type
        vehicle['name'] = self.name
        vehicle['id'] = self.id
        
        vehicle['lat_maneuver_intention'] = self.lat_maneuver_intention
        vehicle['lon_maneuver_intention'] = self.lon_maneuver_intention

        vehicle['lat_maneuver_probs'] = self.lat_maneuver_probs.tolist()                                        
        vehicle['lon_maneuver_probs'] = self.lon_maneuver_probs.tolist()
                
        vehicle['lane_deviation']= self.lane_deviation.tolist()\
                            if self.lane_deviation is not None\
                            else None   
                            
        vehicle['lane']=self.lane.tolist()\
                            if self.lane is not None\
                            else None
                                                
        vehicle['historical_trajectory'] = self.historical_trajectory.tolist()
        vehicle['future_trajectory'] = self.future_trajectory.tolist()
        
        return vehicle
    
    def from_dict(self, v:Dict)->NoReturn:
        
        self.type = v['type']
        self.name = v['name']
        self.id   = v['id']
        
        self.lat_maneuver_intention = v['lat_maneuver_intention']
        self.lon_maneuver_intention = v['lon_maneuver_intention']

        self.lat_maneuver_probs = np.asarray(v['lat_maneuver_probs'])
        self.lon_maneuver_probs = np.asarray(v['lon_maneuver_probs'])

        self.historical_trajectory = np.asarray(v['historical_trajectory'])
        self.future_trajectory = np.asarray(v['future_trajectory'])

        self.lane = np.asarray(v['lane'])
        self.lane_deviation = np.asarray(v['lane_deviation'])
        

class InteractiveGraph(nx.Graph):

    def __init__(self, interaction_field:float, name:str="", **kargs):
        """
            Spatial Interaction Graph definition

        Args:
            interaction_field (float): [radius of the interaction field in meters]
            name (str, optional): [name of the graph]. Defaults to "".
        
        Structure:
            + nodes: 
                - (1) Target Vehicle 
                - (2) Surrounding Vehicles
            + edges
                - (1) Spatial Edges: [same time slice]
        """
        super(InteractiveGraph, self).__init__(**kargs)   

        self.name = name
        self.interaction_field:float = interaction_field
        
        self.target_vehicle:int=-1     
        self.ego_vehicle:int=-1

    def add_vehicle(self, v:Vehicle, force:bool=False):
        """
        add a new node to the graph

        Args:
            v (Vehicle): [new vehicle]
        """
        
        if v is None:
            raise ValueError('None object not allowed!')

        if v.id in self.nodes:
            raise RuntimeError(f'Vehicle ID is already in the graph (id:{v.id})')
            
        if (v.type == Vehicle.Category.TARGET_VEHICLE) and\
           (not force) and\
           (self.target_vehicle>=0):
            
            raise RuntimeError(("There is already one target vehicle in the graph."
                                f"(target_vehicle:{self.target_vehicle}).\n"
                                "You can set `force` to True, to override the current"
                                " target vehicle."))
            
        if v.type == Vehicle.Category.TARGET_VEHICLE:
            
            if self.target_vehicle>=0:
                self.nodes[self.target_vehicle].get("content").type=Vehicle.Category.GENERIC_VEHICLE
            
            self.target_vehicle = v.id
        
        if v.type == Vehicle.Category.EGO_VEHICLE:
            self.ego_vehicle = v.id           
            
        self.add_node(v.id, 
                      content=v,
                      position=v.historical_trajectory[-1][:2])
    
    def update_keys(self):
        """
            change the target_vehicle id to 0
            and map the remaining vehicles to start from 1
        """
        nodes_id = np.fromiter(self.nodes.keys(), dtype=int)

        map_ids = {self.target_vehicle:0}
        
        n_ids = nodes_id[nodes_id!=self.target_vehicle]
        map_ids.update({n:i+1 for i, n in enumerate(n_ids)})

        _nodes = []
        for node in self.nodes.keys():
            v = self.nodes[node].get("content")
            v.id = map_ids[v.id]
            _nodes.append(v)
        _name = self.name
        self.clear()
        self.name = _name
        
        self.target_vehicle=-1
        for node in _nodes:
            self.add_vehicle(node)

        

    def build_weighted_edges(self):
        edges = []

        dist_func = lambda x, y: np.sqrt(np.power(x-y, 2).sum())

        #visited={}
        for node_i  in self.nodes:
            for node_j in self.nodes:
                p_i = self.nodes[node_i].get("position")
                p_j = self.nodes[node_j].get("position")
                d_ij = dist_func(p_i, p_j)
                edges.append([node_i, node_j, d_ij])
        
        self.add_weighted_edges_from(edges)

    def set_target_vehicle(self, id:int, force:bool=False):
        """
        define a target vehicle to the interactive graph

        Args:
            id (int): [id of the new target vehicle]
            force (bool, optional): [force a change of target vehicle, if 
                    there is already one in the graph]. Defaults to False.
        """
        if id not in self.nodes:
            raise KeyError(f"Vehicle ID not found. (id:{id})")
        
        if (not force) and (self.target_vehicle>=0):
            raise RuntimeError(("There is already one target vehicle in the graph."
                                f"(target_vehicle:{self.target_vehicle}).\n"
                                "You can set `force` to True, to override the current"
                                " target vehicle."))
        
                  
        if force and (self.target_vehicle>=0):
            self.nodes[self.target_vehicle].get("content").type = Vehicle.Category.GENERIC_VEHICLE
        
                       
        self.nodes[id].get("content").type = Vehicle.Category.TARGET_VEHICLE
        self.target_vehicle = id       
        
    def set_ego_vehicle(self, id:int, force:bool=False):
        """
        define a ego vehicle to the interactive graph

        Args:
            id (int): [id of the new ego vehicle]
            force (bool, optional): [force a change of target vehicle, if 
                    there is already one in the graph]. Defaults to False.
        """
        if id not in self.nodes:
            raise KeyError(f"Vehicle ID not found. (id:id)")
        
        if (not force) and (self.ego_vehicle>=0):
            raise RuntimeError(("There is already one ego vehicle in the graph."
                                f"(ego_vehicle:{self.ego_vehicle}).\n"
                                "You can set `force` to True, to override the current"
                                " ego vehicle."))
        
        if force and (self.ego_vehicle>=0):
            self.nodes[self.ego_vehicle].get("content").type = Vehicle.Category.GENERIC_VEHICLE
        
        
        self.nodes[id].get("content").type=Vehicle.Category.EGO_VEHICLE
        self.ego_vehicle = id
          
    def get_edges_using_interaction_field(self):
        """
            return a list of edges from the graph:
        
        - example
            [(v0, v1), (v0,v2), (v2,v3)]
        
        - obs:
            the weight of any edge is lower or equal to the
                radius of the interaction field
        """

        if (self.target_vehicle < 0) and\
            (self.target_vehicle not in self.nodes):
                raise RuntimeError('Target vehicle not found!')
        
        r = self.interaction_field

        nodes_ids = np.sort([i for i in self.nodes])
        adj_matrix = nx.to_numpy_array(self, nodelist=nodes_ids)
        
        map_ids = {i:v for i, v in enumerate(nodes_ids)}

        edges = [[map_ids[i], map_ids[j]]
                    for i, v in enumerate(adj_matrix) 
                    for j in np.where(v<r)[0]
                    if i!=j
                ]

        #target vehicle self-connection 
        edges.append([0,0])

        edges = np.asarray(edges)

        return edges, nodes_ids, adj_matrix
        
    
    def get_graph_from_interaction_field(self, max_neighbors:int=-1):
        
        if (self.target_vehicle < 0) and\
            (self.target_vehicle not in self.nodes):
                raise RuntimeError('Target vehicle not found!')
        
        nodes_ids = np.sort([n for n in self.nodes])
        adj_matrix = nx.to_numpy_array(self, nodelist=nodes_ids)
        
        idx_target = np.where(nodes_ids==self.target_vehicle)[0][0]
        
        target_to_neighbors = adj_matrix[idx_target]
        idx_neighbors = np.where(target_to_neighbors<=self.interaction_field)[0]

        if (max_neighbors>0) and (len(idx_neighbors) > max_neighbors):
            target_to_neighbors = target_to_neighbors[idx_neighbors]
            sorted_idx = np.argsort(target_to_neighbors)[:max_neighbors]
            idx_neighbors = idx_neighbors[sorted_idx]

        idx_neighbors = nodes_ids[idx_neighbors]


        _graph = InteractiveGraph(
            interaction_field=self.interaction_field,
            name=self.name
        )
        for idn in idx_neighbors:
            _graph.add_vehicle(self.nodes[idn].get("content"))
        _graph.build_weighted_edges()
        return _graph
        
        
            
                
