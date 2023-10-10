import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt

def _get_translation_matrix_2d(
    T:np.ndarray
)->np.ndarray:
    """
    return the homogenous translation matrix

    Args:
        T (np.ndarray): [translation offset]

    Returns:
        np.ndarray: [homogenous matrix]
    """
    H = [[1, 0, T[0]],
         [0, 1, T[1]],
         [0, 0,   1]]
    
    return np.asarray(H)

def _get_rotation_matrix_2d(
    rot:float
)->np.ndarray:
    """
    return the homogenous rotation matrix

    Args:
        rot (float): [rotation angle in radian]

    Returns:
        np.ndarray: [homogenous matrix]
    """
    H = [[np.cos(rot), -np.sin(rot), 0],
         [np.sin(rot),  np.cos(rot), 0],
         [0,            0,           1]]
    
    return np.asarray(H)

def _get_tranform_matrix_2d(
    T:np.ndarray, 
    rot:float
)->np.ndarray:
    """
    return the homogenous transformation matrix
    - rotation followed by translation

    Args:
        T (np.ndarray): [translation offset]
        rot (float): [rotation angle]

    Returns:
        np.ndarray: [homogenous matrix]
    """
    
    H = [[np.cos(rot), -np.sin(rot), T[0]],
         [np.sin(rot),  np.cos(rot), T[1]],
         [0          ,  0          ,   1]]
    
    return np.asarray(H)

def _get_translation_rotation_matrix_2d(
    T:np.ndarray,
    rot:float,
)->np.ndarray:
    """
    return a transformation matrix with translation before 
    rotation
    
    - translation followed by rotation

    Args:
        T (np.ndarray): [description]
        rot (float): [description]

    Returns:
        np.ndarray: [description]
    """
    h, k = T[0], T[1]
    a, b = np.cos(rot), np.sin(rot)
    H = [[a, -b, h*a - k*b],
         [b,  a, h*b + k*a],
         [0,  0,     1]]
    return np.asarray(H)
    
def to_origin(seq:np.ndarray)->np.ndarray:
    """
    transform a sequence to the origin
    
    translation based on the first point
    rotation angle based on the first two points

    Args:
        seq (np.ndarray): [sequence]

    Returns:
        np.ndarray: [sequence after transformation]
    """
    T = seq[0, :]
    
    yy= seq[1,1] - seq[0,1]
    xx= seq[1,0] - seq[0,0] 
    
    ang = -np.arctan2(yy, xx)
    
    # seq = seq - seq[0]
    # H = _get_tranform_matrix_2d(T=-T, rot=ang)
    # H = _get_translation_matrix_2d(T=-T)
    H = _get_translation_rotation_matrix_2d(T=-T, rot=ang)
    
    new_seq = [np.dot(H, [t[0], t[1], 1])[:2] 
                for t in seq]
    
    return np.asarray(new_seq)


def rotate(seq:np.ndarray, rot:float)->np.ndarray:
    """
    apply rotation to a sequence (x,y) by `rot` angle 

    Args:
        seq (np.ndarray): [sequence]
        rot (float): [angle in radians]

    Returns:
        np.ndarray: [sequence after rotation]
    """
    
    H = _get_rotation_matrix_2d(rot=rot)
    
    new_seq = [np.dot(H, [t[0], t[1], 1])[:2] 
               for t in seq]
    
    return np.asarray(new_seq)

def translate(seq:np.ndarray, T:List[float])->np.ndarray:
    """
    apply translation to a sequence by T offset (x, y)

    Args:
        seq (np.ndarray): [sequence]
        T (List[float]): [offset [x,y]]

    Returns:
        np.ndarray: [sequence after translation]
    """
    
    H = _get_translation_matrix_2d(T=T)
    
    new_seq = [np.dot(H, [t[0], t[1], 1])[:2] 
               for t in seq]
    
    return np.asarray(new_seq)

def transform(seq:np.ndarray, T:List[float]=[0.,0.], rot:float=0.0) -> np.ndarray:

    H = _get_translation_rotation_matrix_2d(T=T, rot=rot)

    new_seq = [np.dot(H, [t[0], t[1], 1])[:2] 
                for t in seq]
    
    return np.asarray(new_seq)


# if __name__ == "__main__":
    
#     # traj = np.array([[1,1], [2,2], [3,3], [4,4]])
    
#     traj = [[ 410.87078007, 1401.21125203],
#             [ 410.76432241, 1403.78991103],
#             [ 410.72697762, 1405.88802669],
#             [ 410.63259122, 1406.79628075],
#             [ 410.47823643, 1408.90946286],
#             [ 410.44470546, 1411.20980469],
#             [ 410.29747282, 1412.31705638],
#             [ 410.31385856, 1414.47635935],
#             [ 410.1726645,  1416.6332743 ],
#             [ 410.12494019, 1417.76765362],
#             [ 410.03145995, 1419.82559962],
#             [ 409.9389752,  1420.84521554],
#             [ 409.84248177, 1421.81590743],
#             [ 409.76496981, 1422.86454565],
#             [ 409.68645926, 1423.9014871 ],
#             [ 409.59702301, 1424.90003825],
#             [ 409.59702301, 1424.90003825],
#             [ 409.59702301, 1424.90003825],
#             [ 409.59702301, 1424.90003825],
#             [ 409.59702301, 1424.90003825],
#             [ 409.59702301, 1424.90003825],
#             [ 409.04383731, 1429.57530583],
#             [ 408.81326658, 1431.47923322],
#             [ 408.5176883,  1433.14456255],
#             [ 408.43727554, 1434.2323527 ],
#             [ 408.16525355, 1436.04493048],
#             [ 408.02360323, 1437.04849905],
#             [ 407.77012929, 1438.71722815],
#             [ 407.65233275, 1439.65609623],
#             [ 407.41084684, 1441.48214251],
#             [ 407.15405185, 1443.28463816],
#             [ 407.06001116, 1444.33962021],
#             [ 406.87608785, 1446.13699087],
#             [ 406.70379445, 1448.04110134],
#             [ 406.62255443, 1449.00879015],
#             [ 406.52505973, 1450.93426678],
#             [ 406.40104574, 1452.78468822],
#             [ 406.33977058, 1453.66723209],
#             [ 406.29172279, 1454.67856152],
#             [ 406.23452524, 1455.62322371],
#             [ 406.1782417,  1456.50949874],
#             [ 406.04272875, 1458.39536529],
#             [ 405.97064253, 1460.3145961 ],
#             [ 405.92566453, 1461.2759302 ],
#             [ 405.87844705, 1462.19002564],
#             [ 405.82417445, 1463.19158418],
#             [ 405.68037005, 1465.94135037],
#             [ 405.66057286, 1466.92111185],
#             [ 405.57313168, 1469.6666213 ],
#             [ 405.51779217, 1470.57212204]]
    
#     traj = np.asarray(traj)
#     # traj2 = traj - traj[0, :]
#     m = to_origin(seq=traj)
    
#     print(traj[:5, :])
#     print(m[:5,:])
#     plt.figure(1)
#     plt.plot(traj[:, 0], traj[:, 1], 'r.')
#     # plt.plot(traj2[:, 0], traj2[:, 1], 'b.')
#     plt.plot(m[:, 0], m[:,1], 'g*')
#     plt.show()