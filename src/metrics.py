import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Union
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import entropy

class HMMMetrics:
    """Metrics for evaluating HMM performance."""
    
    @staticmethod
    def compute_state_assignment_metrics(
        true_states: np.ndarray,
        pred_states: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute state assignment metrics.

        Parameters
        ----------
        true_states : np.ndarray
            True state assignments
        pred_states : np.ndarray
            Predicted state assignments

        Returns
        -------
        Dict[str, float]
            Dictionary containing ARI and NMI scores
        """
        # Convert inputs to numpy arrays if they aren't already
        true_states = np.asarray(true_states)
        pred_states = np.asarray(pred_states)
        
        # Validate shapes
        if true_states.shape != pred_states.shape:
            raise ValueError("True and predicted states must have the same shape")
        
        # Compute metrics
        ari = adjusted_rand_score(true_states, pred_states)
        nmi = normalized_mutual_info_score(true_states, pred_states)
        
        return {"ari": ari, "nmi": nmi}
    
    @staticmethod
    def compute_log_likelihood(
        data: torch.Tensor,
        log_likelihood: float
    ) -> float:
        """
        Compute log likelihood of data under the model.

        Parameters
        ----------
        data : torch.Tensor
            Input data
        log_likelihood : float
            Raw log likelihood value

        Returns
        -------
        float
            Log likelihood value
        """
        # Validate inputs
        if not isinstance(data, torch.Tensor):
            raise ValueError("Data must be a torch.Tensor")
        if not isinstance(log_likelihood, (int, float)):
            raise ValueError("Log likelihood must be a number")
        
        # Return 0 for positive log likelihoods (invalid)
        if log_likelihood > 0:
            return 0.0
        
        return float(log_likelihood)
    
    @staticmethod
    def compute_transition_matrix_error(
        true_trans: np.ndarray,
        est_trans: np.ndarray
    ) -> float:
        """
        Compute error between true and estimated transition matrices.

        Parameters
        ----------
        true_trans : np.ndarray
            True transition matrix
        est_trans : np.ndarray
            Estimated transition matrix

        Returns
        -------
        float
            KL divergence between matrices
        """
        # Validate inputs
        if not isinstance(true_trans, np.ndarray) or not isinstance(est_trans, np.ndarray):
            raise ValueError("Transition matrices must be numpy arrays")
        if true_trans.shape != est_trans.shape:
            raise ValueError("Transition matrices must have the same shape")
        
        # Validate probability distributions
        if not np.allclose(true_trans.sum(axis=1), 1.0) or not np.allclose(est_trans.sum(axis=1), 1.0):
            raise ValueError("Transition matrices must be valid probability distributions")
        
        # Compute KL divergence
        kl_div = 0.0
        for i in range(true_trans.shape[0]):
            for j in range(true_trans.shape[1]):
                if true_trans[i, j] > 0 and est_trans[i, j] > 0:
                    kl_div += true_trans[i, j] * np.log(true_trans[i, j] / est_trans[i, j])
        
        return kl_div
    
    @staticmethod
    def compute_anomaly_detection_metrics(
        true_anomalies: List[int],
        detected_anomalies: List[Tuple[int, float, float]],
        sequence_length: int
    ) -> Dict[str, float]:
        """
        Compute metrics for anomaly detection performance.

        Parameters
        ----------
        true_anomalies : List[int]
            Indices of true anomalies
        detected_anomalies : List[Tuple[int, float, float]]
            Detected anomalies with scores and confidence
        sequence_length : int
            Total length of sequence

        Returns
        -------
        Dict[str, float]
            Dictionary containing precision, recall, and F1 score
        """
        # Handle empty lists
        if not true_anomalies and not detected_anomalies:
            return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
        if not true_anomalies:
            return {"precision": 1.0, "recall": 0.0, "f1_score": 0.0}
        if not detected_anomalies:
            return {"precision": 1.0, "recall": 0.0, "f1_score": 0.0}
        
        # Extract indices from detected anomalies
        detected_indices = set(x[0] for x in detected_anomalies)
        true_indices = set(true_anomalies)
        
        # Validate indices
        if max(true_indices) >= sequence_length or max(detected_indices) >= sequence_length:
            raise ValueError("Anomaly indices must be less than sequence length")
        
        # Compute metrics
        true_positives = len(true_indices.intersection(detected_indices))
        false_positives = len(detected_indices - true_indices)
        false_negatives = len(true_indices - detected_indices)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

def compute_state_assignment_metrics(true_states: torch.Tensor, pred_states: torch.Tensor) -> Dict[str, float]:
    """
    Compute state assignment metrics between true and predicted states.
    
    Parameters
    ----------
    true_states : torch.Tensor
        True state assignments
    pred_states : torch.Tensor
        Predicted state assignments
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing ARI and NMI scores
        
    Raises
    ------
    ValueError
        If inputs have different shapes
    """
    if true_states.shape != pred_states.shape:
        raise ValueError("True and predicted states must have the same shape")
    
    # Move tensors to CPU and convert to numpy for sklearn metrics
    true_states_np = true_states.cpu().numpy()
    pred_states_np = pred_states.cpu().numpy()
    
    # Compute metrics
    ari = adjusted_rand_score(true_states_np, pred_states_np)
    nmi = normalized_mutual_info_score(true_states_np, pred_states_np)
    
    return {"ari": ari, "nmi": nmi}

def compute_transition_matrix_error(true_trans: torch.Tensor, est_trans: torch.Tensor) -> float:
    """
    Compute error between true and estimated transition matrices.
    
    Parameters
    ----------
    true_trans : torch.Tensor
        True transition matrix
    est_trans : torch.Tensor
        Estimated transition matrix
        
    Returns
    -------
    float
        Mean squared error between matrices
        
    Raises
    ------
    ValueError
        If matrices have different shapes or are not valid probability distributions
    """
    # Validate shapes
    if true_trans.shape != est_trans.shape:
        raise ValueError(f"Transition matrices must have the same shape. Got {true_trans.shape} and {est_trans.shape}")
    
    # Validate probability distributions
    if not torch.allclose(true_trans.sum(dim=1), torch.ones_like(true_trans[0])):
        raise ValueError("True transition matrix rows must sum to 1")
    if not torch.allclose(est_trans.sum(dim=1), torch.ones_like(est_trans[0])):
        raise ValueError("Estimated transition matrix rows must sum to 1")
    
    # Compute MSE
    error = torch.mean((true_trans - est_trans) ** 2)
    return error.item()

def determine_optimal_sampling_ratio(data: torch.Tensor, states: torch.Tensor, target_memory_gb: float = 1.0) -> float:
    """
    Dynamically determine the optimal sampling ratio based on state distribution and memory constraints.
    
    Parameters
    ----------
    data : torch.Tensor
        Input data
    states : torch.Tensor
        State assignments
    target_memory_gb : float
        Target memory usage in GB
        
    Returns
    -------
    float
        Optimal sampling ratio between 0 and 1
        
    Raises
    ------
    ValueError
        If inputs are empty or invalid
    """
    if len(data) == 0 or len(states) == 0:
        raise ValueError("Data and states cannot be empty")
    
    if data.shape[0] != states.shape[0]:
        raise ValueError("Data and states must have the same length")
        
    # Get state distribution
    unique_states = torch.unique(states)
    state_counts = torch.bincount(states, minlength=len(unique_states))
    state_probs = state_counts.float() / len(states)
    
    # Calculate minimum samples needed per state for representation
    min_samples_per_state = 100  # Minimum samples to maintain state characteristics
    total_min_samples = min_samples_per_state * len(unique_states)
    
    # Calculate memory per sample (in GB)
    memory_per_sample = (data.element_size() * data.shape[1] + states.element_size()) / (1024**3)
    memory_per_sample *= 1.2  # Add 20% overhead for PyTorch tensor metadata
    
    # Calculate maximum samples based on target memory
    max_samples = int(target_memory_gb / memory_per_sample)
    
    # Calculate ratio based on minimum samples needed
    ratio_from_min = total_min_samples / len(data)
    
    # Calculate ratio based on memory constraints
    ratio_from_memory = max_samples / len(data)
    
    # Take the larger ratio to ensure both representation and memory constraints are met
    optimal_ratio = max(ratio_from_min, ratio_from_memory)
    
    # Ensure ratio is between 0.1 and 1.0
    optimal_ratio = max(0.1, min(1.0, optimal_ratio))
    
    return optimal_ratio 