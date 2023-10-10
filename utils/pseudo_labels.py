import os
import sys
from pathlib import Path
import numpy as np
from typing import Tuple, List
from copy import deepcopy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.layers.graph_maneuver import ManeuverPrediction

def get_pseudo_label(
    model:ManeuverPrediction,
    X:Tuple,
    mode:str,
    threshold:float=0.5,
)->Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, List, List],
    Tuple[np.ndarray, np.ndarray]
]:

    _y_pred_lat, _y_pred_lon =\
        model.predict(
            X, 
            workers=8, 
            use_multiprocessing=True,
            verbose=1
        )
    
    if mode == "soft":
        _y_pred_lat = np.round(_y_pred_lat, 3)
        _y_pred_lon = np.round(_y_pred_lon, 3)
    elif mode == "hard":
        _y_pred_lat = (_y_pred_lat>=threshold).astype(dtype=int)
        _y_pred_lon = (_y_pred_lon>=threshold).astype(dtype=int)
    else:
        raise NotImplementedError(f"pseudo-label mode not implemented! {mode}")
    
    index = np.where(_y_pred_lat>threshold)[0]
    
    _y_pred_lat = _y_pred_lat[index]
    _y_pred_lon = _y_pred_lon[index]


    _x = []
    for x in X:
        if isinstance(x, np.ndarray):
            _x.append(x[index])
        else:
            _x.append([x[i] for i in index])

    return _x, [_y_pred_lat, _y_pred_lon]

