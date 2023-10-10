
import numpy as np
from typing import Tuple, List, Any

def train_test_split(
    X:np.ndarray,
    Y_lat: np.ndarray,
    Y_lon: np.ndarray,
    test_size:float=0.2,
    random_state:int=42,
    stratify:List[int]=None,
    shuffle:bool=True,
    encoded:bool=False,
)->Tuple[Tuple[np.ndarray, np.ndarray],
         Tuple[np.ndarray, np.ndarray],
         Tuple[np.ndarray, np.ndarray]]:
    
    x_w = X.shape[0]
    ylat_w = Y_lat.shape[0]
    ylon_w = Y_lon.shape[0]
    
    assert (x_w == ylat_w) and (x_w == ylon_w),\
        ("[train_test_split][ERROR] mismatch between array shapes!"
         f" X:{X.shape} | Y_lat:{Y_lat.shape} | Y_lon:{Y_lon.shape}")
    
           
        
    if stratify is not None:
       
        dec_y_lat = np.argmax(Y_lat, axis=1) if encoded else Y_lat
                            
        idx = {s:np.where(dec_y_lat==s)[0] for s in stratify}
        
        _X_train, _X_test = [], []
        _YLAT_train, _YLAT_test = [], []
        _YLON_train, _YLON_test = [], []
        
        for s,v in idx.items():
            size = len(v)
            train_len = int(size*(1-test_size))
            test_len  = size - train_len
                        
            if shuffle:
                idx_train = np.random.choice(v, train_len, replace=False)
                idx_test  = np.setdiff1d(v, idx_train, assume_unique=True)
            else:
                idx_train = v[:train_len]
                idx_test  = v[train_len:]
                
            _X_train.append(X[idx_train])
            _X_test.append(X[idx_test])
            
            _YLAT_train.append(Y_lat[idx_train])   
            _YLAT_test.append(Y_lat[idx_test])
            
            _YLON_train.append(Y_lon[idx_train])
            _YLON_test.append(Y_lon[idx_test])
            
        _X_train = np.concatenate(_X_train, axis=0)
        _X_test = np.concatenate(_X_test, axis=0)
        
        _YLAT_train = np.concatenate(_YLAT_train, axis=0)
        _YLAT_test = np.concatenate(_YLAT_test, axis=0)
        
        _YLON_train = np.concatenate(_YLON_train, axis=0)
        _YLON_test = np.concatenate(_YLON_test, axis=0)
        
        return ((_X_train,_X_test),\
                (_YLAT_train,_YLAT_test), 
                (_YLON_train,_YLON_test))

    else:
        train_len = int(x_w*(1-test_size))
        test_len  = x_w - train_len
    
        idx = range(0, x_w)
        
        if shuffle:
            idx_train = np.random.choice(idx, train_len, replace=False)
            idx_test  = np.setdiff1d(idx, idx_train, assume_unique=True)
        else:
            idx_train = range(0, train_len)
            idx_test  = range(train_len, x_w)
                    
        _X = (X[idx_train], X[idx_test])
        _YLAT = (Y_lat[idx_train], Y_lat[idx_test])
        _YLON = (Y_lon[idx_train], Y_lon[idx_test])
        
        return (_X, _YLAT, _YLON)
    
    
def dataset_split(
    X:Any,
    Y:Any,
    seqs:np.ndarray=None,
    test_size:float=0.2,
    random_state:int=42,
    stratify:List[int]=None,
    shuffle:bool=True,
    encoded:bool=False
):
    x_size = np.shape(X[0])[0]
    y_size = np.shape(Y[0])[0]

    seqs_size = len(seqs) if seqs is not None else x_size
    
    assert (x_size== y_size) and (x_size==seqs_size)
    
    train = None
    test  = None
    
    
    if stratify is not None:
        Y_lat = Y[1]
        
        dec_y_lat = np.argmax(Y_lat, axis=1) if encoded else Y_lat
                            
        idx = {s:np.where(dec_y_lat==s)[0] for s in stratify}
        
        idx_train = np.array([])
        idx_test = np.array([])
        
        for s,v in idx.items():
            size = len(v)
            train_len = int(size*(1-test_size))
            test_len  = size - train_len
                        
            if shuffle:
                idx_train = np.append(idx_train, np.random.choice(v, train_len, replace=False))
                idx_test = np.append(idx_test, np.setdiff1d(v, idx_train, assume_unique=True))
            else:
                idx_train = np.append(idx_train,v[:train_len])
                idx_test = np.append(idx_test, v[train_len:])
        else:
            idx_test = np.array(idx_test, dtype=int)
            idx_train = np.array(idx_train, dtype=int)
  
        
    else:
        train_len = int(x_size*(1-test_size))
        test_len  = x_size - train_len
    
        idx = range(0, x_size)
        
        if shuffle:
            idx_train = np.random.choice(idx, train_len, replace=False)
            idx_test  = np.setdiff1d(idx, idx_train, assume_unique=True)
        else:
            idx_train = range(0, train_len)
            idx_test  = range(train_len, x_size)
        
       
    x_test = get_item_from_ragged_nested_tuple(X, idx_test)
    y_test = get_item_from_ragged_nested_tuple(Y, idx_test)
        
    x_train = get_item_from_ragged_nested_tuple(X, idx_train)
    y_train = get_item_from_ragged_nested_tuple(Y, idx_train)

    if seqs is not None:
        seqs_test = seqs[idx_test]
        seqs_train = seqs[idx_train]
    else:
        seqs_test = None
        seqs_train = None
        
    test = (x_test, y_test, seqs_test)
    train = (x_train, y_train, seqs_train)

    return (train, test)

def get_item_from_ragged_nested_tuple(
    X:Any,
    indexes:List[int]
):
    result = []
    
    for xi in X:
        if isinstance(xi, List):
            slice = [xi[idx] for idx in indexes]
        elif isinstance(xi, np.ndarray):
            slice = xi[indexes]
        
        result.append(slice)
    
    return result
            
def split_batches(
    X:Any, 
    Y:Any,
    batch_size:int,
    shuffle:bool=True,
):
    data_size = np.shape(X[0])[0]
    # split_size = np.ceil(data_size/batch_size).astype(np.int)
    
    data_idx = np.arange(0,data_size)
    if shuffle:
        np.random.shuffle(data_idx)
        
    data_idx = [data_idx[i:i+batch_size] 
                for i in range(0, data_size, batch_size)]
    
    batches =\
        [(get_item_from_ragged_nested_tuple(X, idx),
          get_item_from_ragged_nested_tuple(Y, idx))
         for idx in data_idx]
    
    return batches
    
    
def split_data(
    X:Tuple, 
    Y:Tuple,
    seqs:np.ndarray=None,
    test_rate:float=0.2, 
    val_rate:float=0.2,
):

    if np.isclose(test_rate, 0.0):
        X_train, Y_train = X, Y
        X_test, Y_test, seqs_test = None, None, None
    else:
        (X_train, Y_train, seqs_train),\
        (X_test, Y_test, seqs_test)=\
            dataset_split(
                X, 
                Y, 
                seqs=seqs,
                test_size=test_rate, 
                random_state=42,
                stratify=[0, 1, 2, 3, 4],
                encoded=True,
            )
    
    if np.isclose(val_rate, 0):
        X_val, Y_val, seqs_val = None, None, None
    else:
        (X_train, Y_train, seqs_train),\
        (X_val, Y_val, seqs_val)=\
            dataset_split(
                X_train, 
                Y_train,
                seqs=seqs,
                test_size=val_rate, 
                random_state=42,
                stratify=[0, 1, 2, 3, 4],
                encoded=True,
            )
        
    return ((X_train, Y_train, seqs_train),
            (X_val, Y_val, seqs_val), 
            (X_test, Y_test, seqs_test))
