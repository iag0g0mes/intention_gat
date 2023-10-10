import os
import shutil
import tempfile 
import time
from typing import Any, Dict, List, Tuple, NoReturn

import argparse
import configparser
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import pickle as pkl

from argoverse_manager import ArgoverseManager

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

if __name__ == "__main__":
    """Load sequences and save the computed features"""

    print("[Extract Sequences] running.....")

    start = time.time()

    args = parse_arguments()
    
    if not os.path.exists(args.cfg):
        raise FileNotFoundError(("config file not found!"
                                 f"(cfg:{args.cfg})"))

    config = configparser.ConfigParser()
    config.read(args.cfg)    

    for k, v in config['DIRS'].items():
        if not os.path.exists(config['DIRS'][k]):
            raise FileNotFoundError((f"{k} not found! ({k}:{v})"))
    
    for k, v in config['PATHS'].items():
        if not os.path.exists(config['PATHS'][k]):
            raise FileNotFoundError((f"{k} not found! ({k}:{v})"))


    argoverse = ArgoverseManager(
                    root=config['DIRS']['data_dir'],
                    maneuver_path=config['PATHS']['maneuvers_path'],
                    filter=config['PARAMS']['filter']
                )

    
    print("[Extract Sequences] extracting features....")

    argoverse.process(obs_len=config['PARAMS'].getint('obs_len'),
                      mode=config['PARAMS']['mode'],
                      batch_size=config['PARAMS'].getint('batch_size'),
                      save_dir=config['DIRS']['feature_dir'],
                      interaction_field = config['PARAMS'].getint('interaction_field'),
                      max_neighbors = config['PARAMS'].getint('max_neightbors')
                      )

    print("[Extract Sequences] done!")
    print("[Extract Sequences] time: {} minutes".format((time.time() - start)/60.))

