import pytest
import torch
import numpy as np
from src.metrics import (
    compute_state_assignment_metrics,
    compute_transition_matrix_error,
    determine_optimal_sampling_ratio,
    HMMMetrics
)

# Define test order
pytestmark = pytest.mark.order

@pytest.fixture
def device():
    """Get the device to use for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def sample_data(device):
    """Generate sample data and states for testing."""
    torch.manual_seed(42)
    n_samples = 100
    n_features = 2
    n_states = 3
    
    # Generate random data
    data = torch.randn(n_samples, n_features).to(device)
    
    # Generate random state assignments
    true_states = torch.randint(0, n_states, (n_samples,)).to(device)
    pred_states = torch.randint(0, n_states, (n_samples,)).to(device)
    
    return data, true_states, pred_states

@pytest.fixture
def hmm_metrics():
    """Create HMMMetrics instance."""
    return HMMMetrics()

@pytest.mark.order(1)
def test_hmm_metrics_initialization(hmm_metrics):
    """Test HMMMetrics class initialization."""
    assert isinstance(hmm_metrics, HMMMetrics)
    assert hasattr(hmm_metrics, 'compute_state_assignment_metrics')
    assert hasattr(hmm_metrics, 'compute_log_likelihood')
    assert hasattr(hmm_metrics, 'compute_transition_matrix_error')
    assert hasattr(hmm_metrics, 'compute_anomaly_detection_metrics')

@pytest.mark.order(2)
def test_hmm_state_metrics(hmm_metrics, sample_data):
    """Test HMMMetrics state assignment metrics."""
    _, true_states, pred_states = sample_data
    
    # Convert to numpy for HMMMetrics
    true_np = true_states.cpu().numpy()
    pred_np = pred_states.cpu().numpy()
    
    # Test with same states
    metrics = hmm_metrics.compute_state_assignment_metrics(true_np, true_np)
    assert metrics['ari'] == 1.0
    assert metrics['nmi'] == 1.0
    
    # Test with different states
    metrics = hmm_metrics.compute_state_assignment_metrics(true_np, pred_np)
    assert -1.0 <= metrics['ari'] <= 1.0
    assert 0.0 <= metrics['nmi'] <= 1.0
    
    # Test error case
    with pytest.raises(ValueError):
        hmm_metrics.compute_state_assignment_metrics(true_np, true_np[:-1])

@pytest.mark.order(3)
def test_hmm_log_likelihood(hmm_metrics, sample_data):
    """Test HMMMetrics log likelihood computation."""
    data, _, _ = sample_data
    
    # Test valid negative log likelihood
    ll = hmm_metrics.compute_log_likelihood(data, -100.0)
    assert ll == -100.0
    
    # Test invalid positive log likelihood
    ll = hmm_metrics.compute_log_likelihood(data, 100.0)
    assert ll == 0.0
    
    # Test error cases
    with pytest.raises(ValueError):
        hmm_metrics.compute_log_likelihood(np.array([1, 2, 3]), -100.0)  # Wrong data type
    with pytest.raises(ValueError):
        hmm_metrics.compute_log_likelihood(data, "invalid")  # Wrong likelihood type

@pytest.mark.order(4)
def test_hmm_transition_error(hmm_metrics, device):
    """Test HMMMetrics transition matrix error computation."""
    # Create valid transition matrices
    true_matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    
    pred_matrix = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7]
    ])
    
    # Test valid case
    error = hmm_metrics.compute_transition_matrix_error(true_matrix, pred_matrix)
    assert isinstance(error, float)
    assert error >= 0.0
    
    # Test identical matrices
    error = hmm_metrics.compute_transition_matrix_error(true_matrix, true_matrix)
    assert error == 0.0
    
    # Test error cases
    with pytest.raises(ValueError):
        invalid_matrix = np.ones((3, 3))  # Not a probability distribution
        hmm_metrics.compute_transition_matrix_error(true_matrix, invalid_matrix)

@pytest.mark.order(5)
def test_hmm_anomaly_detection(hmm_metrics):
    """Test HMMMetrics anomaly detection metrics."""
    # Test perfect detection
    true_anomalies = [1, 5, 10]
    detected_anomalies = [(1, 0.9, 0.8), (5, 0.8, 0.7), (10, 0.7, 0.6)]
    metrics = hmm_metrics.compute_anomaly_detection_metrics(true_anomalies, detected_anomalies, sequence_length=15)
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1_score'] == 1.0
    
    # Test imperfect detection
    detected_anomalies = [(1, 0.9, 0.8), (5, 0.8, 0.7), (12, 0.7, 0.6)]  # One false positive
    metrics = hmm_metrics.compute_anomaly_detection_metrics(true_anomalies, detected_anomalies, sequence_length=15)
    assert metrics['precision'] < 1.0
    assert metrics['recall'] < 1.0
    assert metrics['f1_score'] < 1.0
    
    # Test edge cases
    metrics = hmm_metrics.compute_anomaly_detection_metrics([], [], sequence_length=15)
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1_score'] == 1.0
    
    metrics = hmm_metrics.compute_anomaly_detection_metrics(true_anomalies, [], sequence_length=15)
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 0.0
    assert metrics['f1_score'] == 0.0
    
    # Test error case
    with pytest.raises(ValueError):
        hmm_metrics.compute_anomaly_detection_metrics([20], [(20, 0.9, 0.8)], sequence_length=15)

@pytest.mark.order(6)
def test_state_assignment_metrics(sample_data):
    """Test standalone state assignment metrics function."""
    _, true_states, pred_states = sample_data
    
    # Test with same states
    metrics = compute_state_assignment_metrics(true_states, true_states)
    assert metrics['ari'] == 1.0
    assert metrics['nmi'] == 1.0
    
    # Test with different states
    metrics = compute_state_assignment_metrics(true_states, pred_states)
    assert -1.0 <= metrics['ari'] <= 1.0
    assert 0.0 <= metrics['nmi'] <= 1.0
    
    # Test error case
    with pytest.raises(ValueError):
        compute_state_assignment_metrics(true_states, true_states[:-1])

@pytest.mark.order(7)
def test_transition_matrix_error(device):
    """Test standalone transition matrix error function."""
    # Create valid transition matrices
    true_matrix = torch.tensor([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ], device=device)
    
    pred_matrix = torch.tensor([
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7]
    ], device=device)
    
    # Test valid case
    error = compute_transition_matrix_error(true_matrix, pred_matrix)
    assert isinstance(error, float)
    assert 0.0 <= error <= 1.0
    
    # Test identical matrices
    error = compute_transition_matrix_error(true_matrix, true_matrix)
    assert error == 0.0
    
    # Test error cases
    with pytest.raises(ValueError):
        wrong_shape = torch.ones((2, 2), device=device)
        compute_transition_matrix_error(true_matrix, wrong_shape)
    
    with pytest.raises(ValueError):
        invalid_prob = torch.ones((3, 3), device=device)
        compute_transition_matrix_error(true_matrix, invalid_prob)

@pytest.mark.order(8)
def test_optimal_sampling_ratio(sample_data):
    """Test optimal sampling ratio determination."""
    data, states, _ = sample_data
    
    # Test valid cases
    ratio = determine_optimal_sampling_ratio(data, states, target_memory_gb=1.0)
    assert isinstance(ratio, float)
    assert 0.1 <= ratio <= 1.0
    
    ratio = determine_optimal_sampling_ratio(data, states, target_memory_gb=0.1)
    assert ratio <= 1.0  # Should be smaller due to memory constraint
    
    # Test error cases
    with pytest.raises(ValueError):
        determine_optimal_sampling_ratio(torch.tensor([]), torch.tensor([]))
    
    with pytest.raises(ValueError):
        determine_optimal_sampling_ratio(data, states[:-1])

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--order-scope=module"]) 