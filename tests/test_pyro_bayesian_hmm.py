import pytest
import torch
import pyro
import sys
from pathlib import Path
import os
import tempfile
import time
import torch.nn as nn

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pyroBayesianHMM import PyroBayesianHMM

@pytest.fixture(scope="function", autouse=True)
def clear_param_store():
    """Clear Pyro's parameter store before and after each test."""
    pyro.clear_param_store()
    yield
    pyro.clear_param_store()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="session")
def device():
    """Get the device to use for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="function")
def sample_data(device):
    """Generate sample data for testing."""
    torch.manual_seed(42)
    n_samples = 2_000  # 2 thousand samples
    n_features = 2
    
    # Create data in chunks to avoid memory issues
    chunk_size = 200  # Chunks for 2k samples
    data_chunks = []
    
    print("\nGenerating 2 thousand samples in chunks...", flush=True)
    for i in range(0, n_samples, chunk_size):
        chunk = torch.randn(min(chunk_size, n_samples - i), n_features, device=device)
        data_chunks.append(chunk)
    
    # Concatenate all chunks
    data = torch.cat(data_chunks, dim=0)
    print(f"Generated data shape: {data.shape}", flush=True)
    return data

@pytest.fixture(scope="function")
def base_model(device):
    """Create a base model instance for testing."""
    model = PyroBayesianHMM(
        n_states=2,  # Reduced number of states
        emission_type="gaussian",
        device=device,
        obs_dim=2,
        hidden_dim=8,  # Reduced hidden dimension
        num_layers=1
    )
    return model

def test_model_initialization(base_model, device):
    """Test model initialization."""
    assert base_model.n_states == 2
    assert base_model.emission_type == "gaussian"
    assert base_model.device == torch.device(device)
    assert base_model.obs_dim == 2
    assert base_model.hidden_dim == 8
    assert base_model.num_layers == 1
    assert not base_model.model_trained
    assert base_model.inference_net.gru.input_size == 2
    assert base_model.inference_net.gru.hidden_size == 8
    assert isinstance(base_model.inference_net.gru, nn.GRU)

def test_basic_training(base_model, sample_data):
    """Test basic model training with chunked processing for large datasets."""
    # Initialize model
    assert not base_model.model_trained
    print("\nTraining Configuration:", flush=True)
    print(f"Model device: {base_model.device}", flush=True)
    print(f"Data shape: {sample_data.shape}", flush=True)
    
    # Train the model using chunked training
    losses = base_model.train_chunked(
        data=sample_data,
        chunk_size=100,  # Chunks for 2k samples
        num_steps=30,    # More steps for 2k samples
        learning_rate=0.0008,  # Slightly lower learning rate for stability
        patience=4,      # Slightly more patience
        min_delta=1e-4,  # Smaller improvement threshold
        batch_chunks=3,  # More chunks per step
        use_stratified=True
    )
    
    # Basic verification
    assert len(losses) > 0
    assert base_model.model_trained
    assert losses[-1] < losses[0]  # Loss should decrease
    
    print("\nTraining completed successfully!", flush=True)

def test_basic_inference(base_model, sample_data):
    """Test basic state inference."""
    print("\nStarting Basic Inference Test:", flush=True)
    print(f"Data shape: {sample_data.shape}", flush=True)
    
    # Train the model first
    print("\nTraining model for inference...", flush=True)
    losses = base_model.train(sample_data, num_steps=10, learning_rate=0.01)
    print(f"Training completed with {len(losses)} steps", flush=True)
    print(f"Final loss: {losses[-1]:.4f}", flush=True)
    
    # Test inference
    print("\nPerforming state inference...", flush=True)
    states = base_model.infer_states(sample_data)
    print(f"States shape: {states.shape}", flush=True)
    print(f"Unique states: {torch.unique(states).tolist()}", flush=True)
    
    # Verify states
    assert isinstance(states, torch.Tensor)
    assert states.shape == (len(sample_data),)
    # Compare device types without index
    expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert states.device.type == expected_device.type
    assert torch.all(states >= 0) and torch.all(states < base_model.n_states)
    print("State verification passed", flush=True)
    
    # Test log probability
    print("\nCalculating log probability...", flush=True)
    log_prob = base_model.infer_log_prob(sample_data)
    print(f"Log probability: {log_prob:.4f}", flush=True)
    
    # Verify log probability
    assert isinstance(log_prob, float)
    assert not torch.isnan(torch.tensor(log_prob))
    print("Log probability verification passed", flush=True)
    
    print("\nBasic inference test completed successfully!", flush=True)
    print("=" * 50, flush=True)  # Visual separator

def test_save_load(base_model, sample_data, device):
    """Test model saving and loading."""
    # Train the model first
    base_model.train(sample_data, num_steps=10, learning_rate=0.01)
    
    # Create temp directory that will be cleaned up
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "test_model.pt")
        
        # Save the model
        base_model.save_model(save_path)
        
        # Clear parameter store before loading
        pyro.clear_param_store()
        
        # Load the model
        loaded_model = PyroBayesianHMM.load_model(save_path, device=device)
        
        # Verify loaded model
        assert loaded_model.n_states == base_model.n_states
        assert loaded_model.emission_type == base_model.emission_type
        assert loaded_model.obs_dim == base_model.obs_dim
        assert loaded_model.device == base_model.device
        assert loaded_model.model_trained
        
        # Verify inference works on loaded model
        states = loaded_model.infer_states(sample_data)
        assert states.shape == (len(sample_data),)

def test_anomaly_detection(base_model, sample_data):
    """Test anomaly detection functionality."""
    # Train the model first
    base_model.train(sample_data, num_steps=10, learning_rate=0.01)
    
    # Test anomaly detection
    anomalies = base_model.detect_anomalies(sample_data, threshold=2.0)
    assert isinstance(anomalies, torch.Tensor)
    assert anomalies.shape[0] == len(sample_data)
    assert anomalies.dtype == torch.bool
    # Compare device types without index
    assert anomalies.device.type == base_model.device.type
    
    # Test with different threshold
    anomalies_strict = base_model.detect_anomalies(sample_data, threshold=1.0)
    assert torch.sum(anomalies_strict) >= torch.sum(anomalies)  # Stricter threshold should find more anomalies

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "test_basic_training", "-s"]) 