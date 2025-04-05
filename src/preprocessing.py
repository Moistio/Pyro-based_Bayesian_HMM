import numpy as np
import torch
import pandas as pd
from typing import Tuple, Optional, Union, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
import numba
from numba import jit, cuda
import cupy as cp

class HMMPreprocessor:
    """Preprocessing utilities for HMM data."""
    
    def __init__(self):
        self.scalers = {}
        
    def normalize_data(
        self,
        data: Union[np.ndarray, torch.Tensor],
        method: str = 'standard',
        feature_wise: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize the input data.
        
        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            Input data to normalize
        method : str
            Normalization method ('standard' or 'minmax')
        feature_wise : bool
            Whether to normalize each feature independently
            
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Normalized data
        """
        if len(data) == 0:
            raise ValueError("Cannot normalize empty data")
            
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            data = data.cpu().numpy()
            
        # Check for invalid values
        if np.any(np.isinf(data)) or np.any(np.isnan(data)):
            raise ValueError("Data contains infinite or NaN values")
            
        if method == 'standard':
            if feature_wise and len(data.shape) > 1:
                # Feature-wise standardization
                mean = np.mean(data, axis=0, keepdims=True)
                std = np.std(data, axis=0, keepdims=True, ddof=1)  # Use ddof=1 for sample std
                # Add epsilon for numerical stability
                normalized = (data - mean) / (std + np.finfo(float).eps)
                # Store parameters
                self.scalers['last'] = {'method': 'standard', 'mean': mean, 'std': std}
            else:
                # Global standardization
                mean = np.mean(data)
                std = np.std(data, ddof=1)  # Use ddof=1 for sample std
                normalized = (data - mean) / (std + np.finfo(float).eps)
                # Store parameters
                self.scalers['last'] = {'method': 'standard', 'mean': mean, 'std': std}
                
        elif method == 'minmax':
            if feature_wise and len(data.shape) > 1:
                # Feature-wise min-max scaling
                min_vals = np.min(data, axis=0, keepdims=True)
                max_vals = np.max(data, axis=0, keepdims=True)
                # Add epsilon for numerical stability
                normalized = (data - min_vals) / (max_vals - min_vals + np.finfo(float).eps)
                # Store parameters
                self.scalers['last'] = {'method': 'minmax', 'min': min_vals, 'max': max_vals}
            else:
                # Global min-max scaling
                min_val = np.min(data)
                max_val = np.max(data)
                normalized = (data - min_val) / (max_val - min_val + np.finfo(float).eps)
                # Store parameters
                self.scalers['last'] = {'method': 'minmax', 'min': min_val, 'max': max_val}
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
        if is_torch:
            normalized = torch.from_numpy(normalized).to(device)
            
        return normalized
    
    def handle_missing_values(
        self,
        data: Union[np.ndarray, torch.Tensor],
        method: str = 'interpolate',
        max_gap: int = 5
    ) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]:
        """
        Handle missing values in the data.
        
        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            Input data with missing values
        method : str
            Method to handle missing values ('interpolate', 'forward', 'mean')
        max_gap : int
            Maximum gap to interpolate across
            
        Returns
        -------
        Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]
            Processed data and mask indicating missing values
        """
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            data = data.cpu().numpy()
            
        if method not in ['interpolate', 'forward', 'mean']:
            raise ValueError(f"Unsupported missing value handling method: {method}")
            
        mask = ~np.isnan(data)
        filled_data = data.copy()
        
        if method == 'interpolate':
            for i in range(data.shape[1] if len(data.shape) > 1 else 1):
                if len(data.shape) > 1:
                    series = data[:, i]
                else:
                    series = data
                    
                valid = ~np.isnan(series)
                if not np.any(valid):
                    raise ValueError(f"Column {i} contains all NaN values")
                    
                indices = np.arange(len(series))
                
                # Only interpolate if gap is not too large
                gaps = np.diff(indices[valid])
                if gaps.size > 0 and np.max(gaps) > max_gap:
                    raise ValueError(f"Gap in column {i} exceeds max_gap of {max_gap}")
                    
                interp_func = np.interp(indices[~valid], indices[valid], series[valid])
                if len(data.shape) > 1:
                    filled_data[~valid, i] = interp_func
                else:
                    filled_data[~valid] = interp_func
                
        elif method == 'forward':
            filled_data = pd.DataFrame(data).ffill().values
            if np.isnan(filled_data).any():
                filled_data = pd.DataFrame(filled_data).bfill().values
            
        elif method == 'mean':
            if len(data.shape) > 1:
                means = np.nanmean(data, axis=0)
                if np.isnan(means).any():
                    raise ValueError("Some columns contain all NaN values")
                for i in range(data.shape[1]):
                    filled_data[:, i][np.isnan(data[:, i])] = means[i]
            else:
                mean_val = np.nanmean(data)
                if np.isnan(mean_val):
                    raise ValueError("Data contains all NaN values")
                filled_data[np.isnan(data)] = mean_val
                
        if is_torch:
            filled_data = torch.from_numpy(filled_data).to(device)
            
        return filled_data, mask
    
    def smooth_data(
        self,
        data: Union[np.ndarray, torch.Tensor],
        window_size: int = 5,
        method: str = 'moving_average'
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Smooth the input data.
        
        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            Input data to smooth
        window_size : int
            Size of the smoothing window
        method : str
            Smoothing method ('moving_average' or 'savgol')
            
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Smoothed data
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")
            
        if method not in ['moving_average', 'savgol']:
            raise ValueError(f"Unsupported smoothing method: {method}")
            
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            data = data.cpu().numpy()
            
        # Check for invalid values
        if np.any(np.isinf(data)) or np.any(np.isnan(data)):
            raise ValueError("Data contains infinite or NaN values")
            
        if method == 'moving_average':
            kernel = np.ones(window_size) / window_size
            if len(data.shape) > 1:
                smoothed = np.apply_along_axis(
                    lambda x: np.convolve(x, kernel, mode='same'), 0, data
                )
            else:
                smoothed = np.convolve(data, kernel, mode='same')
                
        elif method == 'savgol':
            if window_size % 2 == 0:
                window_size += 1  # Savgol requires odd window size
            if len(data.shape) > 1:
                smoothed = np.apply_along_axis(
                    lambda x: signal.savgol_filter(x, window_size, 3), 0, data
                )
            else:
                smoothed = signal.savgol_filter(data, window_size, 3)
                
        if is_torch:
            smoothed = torch.from_numpy(smoothed).to(device)
            
        return smoothed
    
    def segment_sequences(
        self,
        data: Union[np.ndarray, torch.Tensor],
        segment_length: int,
        overlap: int = 0
    ) -> List[Union[np.ndarray, torch.Tensor]]:
        """
        Segment long sequences into shorter ones.
        
        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            Input sequence data
        segment_length : int
            Length of each segment
        overlap : int
            Number of overlapping points between segments
            
        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of segmented sequences
        """
        if segment_length <= 0:
            raise ValueError("Segment length must be positive")
            
        if overlap >= segment_length:
            raise ValueError("Overlap must be less than segment length")
            
        if len(data.shape) > 2:
            raise ValueError("Data must be 1D or 2D")
            
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            data = data.cpu().numpy()
            
        # Calculate step size
        step = segment_length - overlap
        
        # Generate segments
        segments = []
        for i in range(0, len(data) - segment_length + 1, step):
            segment = data[i:i + segment_length]
            if is_torch:
                segment = torch.from_numpy(segment).to(device)
            segments.append(segment)
            
        return segments

# Numba-accelerated preprocessing functions
@jit(nopython=True)
def standardize_numba(data):
    """Numba-accelerated standardization."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std + 1e-8)

@jit(nopython=True)
def minmax_scale_numba(data):
    """Numba-accelerated minmax scaling."""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val + 1e-8)

# CUDA-accelerated preprocessing functions
def standardize_cupy(data):
    """CuPy-accelerated standardization."""
    data_cp = cp.array(data)
    mean = cp.mean(data_cp)
    std = cp.std(data_cp)
    return cp.asnumpy((data_cp - mean) / (std + 1e-8))

def minmax_scale_cupy(data):
    """CuPy-accelerated minmax scaling."""
    data_cp = cp.array(data)
    min_val = cp.min(data_cp)
    max_val = cp.max(data_cp)
    return cp.asnumpy((data_cp - min_val) / (max_val - min_val + 1e-8))

# Numba-accelerated chunking
@jit(nopython=True)
def chunk_data_numba(data, chunk_size):
    """Numba-accelerated data chunking."""
    n_chunks = (len(data) + chunk_size - 1) // chunk_size
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        chunks.append(data[start_idx:end_idx])
    return chunks

# CUDA-accelerated chunking
def chunk_data_cupy(data, chunk_size):
    """CuPy-accelerated data chunking."""
    data_cp = cp.array(data)
    n_chunks = (len(data) + chunk_size - 1) // chunk_size
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        chunk = cp.asnumpy(data_cp[start_idx:end_idx])
        chunks.append(chunk)
    return chunks

def preprocess_data(
    data: Union[np.ndarray, torch.Tensor],
    normalize: bool = True,
    method: str = "standard",
    use_acceleration: bool = True,
    accelerator: Optional[str] = None,
    handle_missing: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Main preprocessing function.
    
    Parameters
    ----------
    data : Union[np.ndarray, torch.Tensor]
        Input data to preprocess
    normalize : bool
        Whether to normalize the data
    method : str
        Normalization method ('standard' or 'minmax')
    use_acceleration : bool
        Whether to use acceleration
    accelerator : Optional[str]
        Type of acceleration to use ('numba' or 'cupy')
    handle_missing : bool
        Whether to handle missing values
        
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Preprocessed data
    """
    is_torch = isinstance(data, torch.Tensor)
    if is_torch:
        device = data.device
        data = data.cpu().numpy()
        
    if handle_missing:
        preprocessor = HMMPreprocessor()
        data, _ = preprocessor.handle_missing_values(data)
        
    if normalize:
        if use_acceleration and accelerator:
            if accelerator == "numba":
                try:
                    data = standardize_numba(data) if method == "standard" else minmax_scale_numba(data)
                except Exception as e:
                    print(f"Numba acceleration failed: {str(e)}, falling back to standard processing")
                    preprocessor = HMMPreprocessor()
                    data = preprocessor.normalize_data(data, method=method)
            elif accelerator == "cupy":
                try:
                    data = standardize_cupy(data) if method == "standard" else minmax_scale_cupy(data)
                except Exception as e:
                    print(f"CuPy acceleration failed: {str(e)}, falling back to standard processing")
                    preprocessor = HMMPreprocessor()
                    data = preprocessor.normalize_data(data, method=method)
            else:
                raise ValueError(f"Unsupported accelerator: {accelerator}")
        else:
            preprocessor = HMMPreprocessor()
            data = preprocessor.normalize_data(data, method=method)
            
    if is_torch:
        data = torch.from_numpy(data).to(device)
        
    return data

def chunk_data(
    data: Union[np.ndarray, torch.Tensor],
    chunk_size: int,
    use_acceleration: bool = True,
    accelerator: Optional[str] = None
) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    Split data into chunks.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
        
    is_torch = isinstance(data, torch.Tensor)
    if is_torch:
        device = data.device
        data = data.cpu().numpy()
        
    if use_acceleration and accelerator:
        if accelerator == "numba":
            try:
                chunks = chunk_data_numba(data, chunk_size)
            except Exception as e:
                print(f"Numba acceleration failed: {str(e)}, falling back to standard processing")
                chunks = [data[i:min(i + chunk_size, len(data))] 
                         for i in range(0, len(data), chunk_size)]
        elif accelerator == "cupy":
            try:
                chunks = chunk_data_cupy(data, chunk_size)
            except Exception as e:
                print(f"CuPy acceleration failed: {str(e)}, falling back to standard processing")
                chunks = [data[i:min(i + chunk_size, len(data))] 
                         for i in range(0, len(data), chunk_size)]
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")
    else:
        chunks = [data[i:min(i + chunk_size, len(data))] 
                 for i in range(0, len(data), chunk_size)]
        
    if is_torch:
        chunks = [torch.from_numpy(chunk).to(device) for chunk in chunks]
        
    return chunks 