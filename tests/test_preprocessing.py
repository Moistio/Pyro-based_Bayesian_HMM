import pytest
import torch
import numpy as np
from src.preprocessing import (
    HMMPreprocessor,
    preprocess_data,
    chunk_data,
    standardize_numba,
    standardize_cupy,
    minmax_scale_numba,
    minmax_scale_cupy,
    chunk_data_numba,
    chunk_data_cupy
)

# Check for optional dependencies
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

@pytest.fixture(scope="session")
def device():
    """Get the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="function")
def preprocessor():
    """Create a fresh preprocessor for each test."""
    return HMMPreprocessor()

@pytest.fixture(scope="function")
def sample_data(device):
    """Generate sample data for testing."""
    torch.manual_seed(42)
    n_samples = 100
    n_features = 2
    # Create data with double precision
    data = torch.randn(n_samples, n_features, dtype=torch.float64, device=device)
    return data

@pytest.fixture(scope="function")
def data_with_missing(device):
    """Generate data with missing values."""
    torch.manual_seed(42)
    data = torch.randn(100, 2, dtype=torch.float64, device=device)
    # Create smaller gaps that don't exceed max_gap
    data[10:12, 0] = float('nan')  # 2-point gap in first column
    data[30:32, 1] = float('nan')  # 2-point gap in second column
    return data

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def test_hmm_preprocessor_initialization():
    """Test HMMPreprocessor initialization."""
    preprocessor = HMMPreprocessor()
    assert hasattr(preprocessor, 'scalers')
    assert isinstance(preprocessor.scalers, dict)

def test_normalize_data(preprocessor, sample_data):
    """Test data normalization methods."""
    # Test standard normalization
    normalized = preprocessor.normalize_data(sample_data, method='standard')
    assert normalized.shape == sample_data.shape
    assert str(normalized.device) == str(sample_data.device)  # Compare device strings
    
    # Test feature-wise normalization (default)
    means = normalized.mean(dim=0)
    stds = normalized.std(dim=0, unbiased=True)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-5)
    
    # Test minmax normalization
    normalized = preprocessor.normalize_data(sample_data, method='minmax')
    assert normalized.shape == sample_data.shape
    assert str(normalized.device) == str(sample_data.device)
    
    # Test minmax range
    min_val = normalized.min()
    max_val = normalized.max()
    assert torch.allclose(min_val, torch.tensor(0.0, device=sample_data.device, dtype=sample_data.dtype), atol=1e-5)
    assert torch.allclose(max_val, torch.tensor(1.0, device=sample_data.device, dtype=sample_data.dtype), atol=1e-5)
    
    # Test global normalization (non-feature-wise)
    normalized = preprocessor.normalize_data(sample_data, method='standard', feature_wise=False)
    assert normalized.shape == sample_data.shape
    assert normalized.device == sample_data.device
    assert normalized.dtype == sample_data.dtype
    
    # Global normalization should have single mean/std
    mean = normalized.mean()
    std = normalized.std(unbiased=True)
    assert torch.allclose(mean, torch.tensor(0.0, dtype=mean.dtype, device=normalized.device), atol=1e-5)
    assert torch.allclose(std, torch.tensor(1.0, dtype=std.dtype, device=normalized.device), atol=1e-5)
    
    # Test with numpy array input
    numpy_data = sample_data.cpu().numpy()
    normalized = preprocessor.normalize_data(numpy_data, method='standard')
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == numpy_data.shape
    assert np.allclose(normalized.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(normalized.std(axis=0, ddof=1), 1, atol=1e-5)
    
    # Test error cases
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        preprocessor.normalize_data(sample_data, method='invalid')
    
    with pytest.raises(ValueError, match="Cannot normalize empty data"):
        preprocessor.normalize_data(torch.tensor([], device=sample_data.device))
    
    # Test with infinity
    inf_data = torch.tensor([[1.0, float('inf')], [2.0, 3.0]], device=sample_data.device)
    with pytest.raises(ValueError, match="Data contains infinite or NaN values"):
        preprocessor.normalize_data(inf_data)

def test_handle_missing_values(preprocessor, data_with_missing):
    """Test missing value handling."""
    # Test interpolation with small gaps
    filled_data, mask = preprocessor.handle_missing_values(
        data_with_missing, 
        method='interpolate', 
        max_gap=5
    )
    assert not torch.isnan(filled_data).any()
    assert mask.shape == data_with_missing.shape
    assert filled_data.device == data_with_missing.device
    assert filled_data.dtype == data_with_missing.dtype
    
    # Convert numpy bool mask to torch bool tensor and move to correct device
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).to(data_with_missing.device)
    assert mask.dtype == torch.bool
    
    # Test forward fill
    filled_data, mask = preprocessor.handle_missing_values(data_with_missing, method='forward')
    assert not torch.isnan(filled_data).any()
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).to(data_with_missing.device)
    assert mask.dtype == torch.bool
    assert filled_data.device == data_with_missing.device
    assert filled_data.dtype == data_with_missing.dtype
    
    # Test mean fill
    filled_data, mask = preprocessor.handle_missing_values(data_with_missing, method='mean')
    assert not torch.isnan(filled_data).any()
    assert mask.shape == data_with_missing.shape
    assert filled_data.device == data_with_missing.device
    assert filled_data.dtype == data_with_missing.dtype
    # Verify mean fill worked correctly
    non_nan_values = data_with_missing[:, 0][~torch.isnan(data_with_missing[:, 0])]
    expected_mean = non_nan_values.mean()
    assert torch.allclose(filled_data[10:12, 0], expected_mean.expand(2))
    
    # Test numpy array input
    numpy_data = data_with_missing.cpu().numpy()
    filled_data, mask = preprocessor.handle_missing_values(numpy_data, method='interpolate')
    assert isinstance(filled_data, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert not np.isnan(filled_data).any()
    assert mask.dtype == bool

def test_smooth_data(preprocessor, sample_data):
    """Test data smoothing methods."""
    # Test moving average
    smoothed = preprocessor.smooth_data(sample_data, window_size=5, method='moving_average')
    assert smoothed.shape == sample_data.shape
    assert str(smoothed.device) == str(sample_data.device)
    assert not torch.isnan(smoothed).any()
    
    # Verify smoothing effect
    original_std = torch.std(sample_data)
    smoothed_std = torch.std(smoothed)
    assert smoothed_std <= original_std  # Smoothing should reduce variation
    
    # Test Savitzky-Golay
    smoothed = preprocessor.smooth_data(sample_data, window_size=5, method='savgol')
    assert smoothed.shape == sample_data.shape
    assert smoothed.device == sample_data.device
    assert smoothed.dtype == sample_data.dtype
    assert not torch.isnan(smoothed).any()
    
    # Test numpy array input
    numpy_data = sample_data.cpu().numpy()
    smoothed = preprocessor.smooth_data(numpy_data, window_size=5, method='moving_average')
    assert isinstance(smoothed, np.ndarray)
    assert smoothed.shape == numpy_data.shape
    assert not np.isnan(smoothed).any()
    
    # Test error cases
    with pytest.raises(ValueError, match="Window size must be positive"):
        preprocessor.smooth_data(sample_data, window_size=0)
    
    with pytest.raises(ValueError, match="Unsupported smoothing method"):
        preprocessor.smooth_data(sample_data, method='invalid')
    
    # Test with infinity
    inf_data = torch.tensor([[1.0, float('inf')], [2.0, 3.0]], device=sample_data.device)
    with pytest.raises(ValueError, match="Data contains infinite or NaN values"):
        preprocessor.smooth_data(inf_data)

def test_segment_sequences(preprocessor, sample_data):
    """Test sequence segmentation."""
    # Test without overlap
    segments = preprocessor.segment_sequences(sample_data, segment_length=20)
    assert len(segments) == 5
    assert all(isinstance(seg, torch.Tensor) for seg in segments)
    assert all(seg.shape == (20, 2) for seg in segments)
    assert all(str(seg.device) == str(sample_data.device) for seg in segments)
    
    # Test with overlap
    segments = preprocessor.segment_sequences(sample_data, segment_length=20, overlap=10)
    assert len(segments) == 9  # More segments due to overlap
    assert all(isinstance(seg, torch.Tensor) for seg in segments)
    assert all(seg.shape == (20, 2) for seg in segments)
    assert all(str(seg.device) == str(sample_data.device) for seg in segments)

@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_numba_acceleration(sample_data):
    """Test Numba-accelerated functions."""
    data_np = sample_data.cpu().numpy()
    
    standardized = standardize_numba(data_np)
    assert standardized.shape == data_np.shape
    assert np.allclose(standardized.mean(), 0, atol=1e-6)
    assert np.allclose(standardized.std(), 1, atol=1e-6)
    
    minmax_scaled = minmax_scale_numba(data_np)
    assert minmax_scaled.shape == data_np.shape
    assert np.all(minmax_scaled >= 0) and np.all(minmax_scaled <= 1)
    
    chunks = chunk_data_numba(data_np, chunk_size=20)
    assert len(chunks) == (len(data_np) + 19) // 20

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_cupy_acceleration(sample_data):
    """Test CuPy-accelerated functions."""
    data_np = sample_data.cpu().numpy()
    
    standardized = standardize_cupy(data_np)
    assert standardized.shape == data_np.shape
    assert np.allclose(standardized.mean(), 0, atol=1e-6)
    assert np.allclose(standardized.std(), 1, atol=1e-6)
    
    minmax_scaled = minmax_scale_cupy(data_np)
    assert minmax_scaled.shape == data_np.shape
    assert np.all(minmax_scaled >= 0) and np.all(minmax_scaled <= 1)
    
    chunks = chunk_data_cupy(data_np, chunk_size=20)
    assert len(chunks) == (len(data_np) + 19) // 20

def test_preprocess_data(sample_data):
    """Test main preprocessing function."""
    # Test with different normalization methods
    processed = preprocess_data(sample_data, normalize=True, method="standard")
    assert isinstance(processed, torch.Tensor)
    assert processed.shape == sample_data.shape
    assert processed.device == sample_data.device
    means = processed.mean(dim=0)
    stds = processed.std(dim=0, unbiased=True)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-5)
    
    processed = preprocess_data(sample_data, normalize=True, method="minmax")
    assert torch.all(processed >= -1e-6) and torch.all(processed <= 1 + 1e-6)
    assert processed.device == sample_data.device
    min_val = processed.min()
    max_val = processed.max()
    assert torch.allclose(min_val, torch.tensor(0.0, dtype=min_val.dtype, device=processed.device), atol=1e-5)
    assert torch.allclose(max_val, torch.tensor(1.0, dtype=max_val.dtype, device=processed.device), atol=1e-5)
    
    # Test with missing values
    data_missing = sample_data.clone()
    data_missing[0, 0] = float('nan')
    processed = preprocess_data(data_missing, normalize=True, handle_missing=True)
    assert not torch.isnan(processed).any()
    assert processed.device == sample_data.device
    
    # Test without normalization
    processed = preprocess_data(sample_data, normalize=False)
    assert torch.equal(processed, sample_data)
    assert processed.device == sample_data.device
    
    # Test error case
    with pytest.raises(ValueError, match="Unsupported accelerator"):
        preprocess_data(sample_data, normalize=True, accelerator='invalid')

def test_chunk_data(sample_data):
    """Test data chunking function."""
    # Test different chunk sizes
    for chunk_size in [10, 20, 50]:
        chunks = chunk_data(sample_data, chunk_size=chunk_size)
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, torch.Tensor) for chunk in chunks)
        assert len(chunks) == (len(sample_data) + chunk_size - 1) // chunk_size
        assert all(chunk.device == sample_data.device for chunk in chunks)
        
        # Test last chunk size
        if len(sample_data) % chunk_size != 0:
            assert chunks[-1].shape[0] == len(sample_data) % chunk_size
        else:
            assert chunks[-1].shape[0] == chunk_size
    
    # Test error cases
    with pytest.raises(ValueError, match="Chunk size must be positive"):
        chunk_data(sample_data, chunk_size=0)
    
    with pytest.raises(ValueError, match="Unsupported accelerator"):
        chunk_data(sample_data, chunk_size=20, accelerator='invalid')

def test_normalize_data_edge_cases(preprocessor, device):
    """Test normalization edge cases."""
    # Test single value data
    single_value = torch.ones(10, 1, device=device)
    normalized = preprocessor.normalize_data(single_value, method='standard')
    assert torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-6)
    assert normalized.device == single_value.device
    
    # Test constant data
    constant_data = torch.full((10, 2), 5.0, device=device)
    normalized = preprocessor.normalize_data(constant_data, method='standard')
    assert torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-6)
    assert normalized.device == constant_data.device
    
    # Test tiny variations
    torch.manual_seed(42)  # For reproducibility
    tiny_var = torch.ones(10, 1, device=device) + torch.randn(10, 1, device=device) * 1e-10
    normalized = preprocessor.normalize_data(tiny_var, method='standard')
    assert not torch.isnan(normalized).any()
    assert normalized.device == tiny_var.device
    
    # Test minmax with tiny range
    tiny_range = torch.ones(10, 1, device=device) + torch.arange(10, device=device).unsqueeze(1) * 1e-10
    normalized = preprocessor.normalize_data(tiny_range, method='minmax')
    assert not torch.isnan(normalized).any()
    assert normalized.device == tiny_range.device
    assert torch.all(normalized >= 0) and torch.all(normalized <= 1)

def test_missing_values_patterns(preprocessor, device):
    """Test handling of different missing value patterns."""
    torch.manual_seed(42)
    
    # Test alternating missing values with small gaps
    alternating = torch.randn(10, 2, dtype=torch.float64, device=device)
    alternating[::3, 0] = float('nan')  # Every third value missing
    filled, mask = preprocessor.handle_missing_values(alternating, method='interpolate', max_gap=2)
    assert not torch.isnan(filled).any()
    assert str(filled.device) == str(alternating.device)
    
    # Test missing values at edges
    edge_missing = torch.randn(10, 2, dtype=torch.float64, device=device)
    edge_missing[0, :] = float('nan')  # First row missing
    edge_missing[-1, :] = float('nan')  # Last row missing
    filled, mask = preprocessor.handle_missing_values(edge_missing, method='forward')
    assert not torch.isnan(filled).any()
    assert str(filled.device) == str(edge_missing.device)

def test_smooth_data_signals(preprocessor, device):
    """Test smoothing with different signal types."""
    # Test sinusoidal signal
    t = torch.linspace(0, 10, 100, device=device)
    sin_wave = torch.sin(t).unsqueeze(1)
    smoothed = preprocessor.smooth_data(sin_wave, window_size=5, method='moving_average')
    assert smoothed.shape == sin_wave.shape
    assert smoothed.device == sin_wave.device
    
    # Test step function
    step = torch.zeros(100, 1, device=device)
    step[50:] = 1.0
    smoothed = preprocessor.smooth_data(step, window_size=5, method='moving_average')
    assert smoothed.shape == step.shape
    assert smoothed.device == step.device
    
    # Test noisy data with both methods
    torch.manual_seed(42)
    noisy = torch.sin(t).unsqueeze(1) + torch.randn(100, 1, device=device) * 0.1
    
    # Test moving average
    smoothed_ma = preprocessor.smooth_data(noisy, window_size=5, method='moving_average')
    assert smoothed_ma.shape == noisy.shape
    assert smoothed_ma.device == noisy.device
    
    # Test savgol
    smoothed_sg = preprocessor.smooth_data(noisy, window_size=5, method='savgol')
    assert smoothed_sg.shape == noisy.shape
    assert smoothed_sg.device == noisy.device

def test_segment_sequences_patterns(preprocessor, device):
    """Test sequence segmentation with different patterns."""
    torch.manual_seed(42)
    
    # Test perfect division
    data = torch.randn(100, 2, dtype=torch.float64, device=device)
    segments = preprocessor.segment_sequences(data, segment_length=20, overlap=0)
    assert len(segments) == 5
    assert all(isinstance(seg, torch.Tensor) for seg in segments)
    assert all(seg.shape == (20, 2) for seg in segments)
    assert all(str(seg.device) == str(data.device) for seg in segments)
    
    # Test with overlap
    segments = preprocessor.segment_sequences(data, segment_length=20, overlap=10)
    assert len(segments) == 9  # Should have more segments due to overlap
    assert all(isinstance(seg, torch.Tensor) for seg in segments)
    assert all(seg.shape == (20, 2) for seg in segments)
    assert all(str(seg.device) == str(data.device) for seg in segments)

def test_segment_sequences_advanced(preprocessor, device):
    """Test advanced sequence segmentation scenarios."""
    # Test with different data shapes and overlap patterns
    data = torch.randn(1000, 3, dtype=torch.float64, device=device)
    
    # Test different segment lengths and overlaps
    configs = [
        (100, 0),    # No overlap
        (100, 50),   # 50% overlap
        (100, 75),   # 75% overlap
        (50, 25),    # Different size with overlap
        (200, 100),  # Larger segments
    ]
    
    for segment_length, overlap in configs:
        segments = preprocessor.segment_sequences(data, segment_length=segment_length, overlap=overlap)
        
        # Check basic properties
        assert all(isinstance(seg, torch.Tensor) for seg in segments)
        assert all(seg.shape[0] == segment_length for seg in segments)
        assert all(seg.device == data.device for seg in segments)
        
        # Verify overlap
        if overlap > 0 and len(segments) > 1:
            first_seg = segments[0]
            second_seg = segments[1]
            assert torch.equal(first_seg[-overlap:], second_seg[:overlap])
        
        # Verify number of segments
        step = segment_length - overlap
        expected_segments = max(1, (len(data) - segment_length) // step + 1)
        assert len(segments) == expected_segments
    
    # Test with exact division
    exact_length = 100
    data_exact = torch.randn(500, 2, dtype=torch.float64, device=device)
    segments = preprocessor.segment_sequences(data_exact, segment_length=exact_length, overlap=0)
    step = exact_length  # no overlap
    expected_segments = max(1, (len(data_exact) - exact_length) // step + 1)
    assert len(segments) == expected_segments
    assert all(seg.shape[0] == exact_length for seg in segments)
    
    # Test with non-exact division
    data_inexact = torch.randn(520, 2, dtype=torch.float64, device=device)
    segments = preprocessor.segment_sequences(data_inexact, segment_length=exact_length, overlap=0)
    step = exact_length  # no overlap
    expected_segments = max(1, (len(data_inexact) - exact_length) // step + 1)
    assert len(segments) == expected_segments
    assert all(seg.shape[0] == exact_length for seg in segments)

def test_preprocess_data_advanced(preprocessor, device):
    """Test advanced preprocessing scenarios."""
    # Test with different data types and configurations
    data = torch.randn(100, 3, dtype=torch.float64, device=device)
    
    # Test normalization methods
    for method in ['standard', 'minmax']:
        # Without acceleration
        processed = preprocess_data(data, normalize=True, method=method, use_acceleration=False)
        assert processed.shape == data.shape
        assert processed.device == data.device
        
        if method == 'minmax':
            assert torch.all(processed >= -1e-6)
            assert torch.all(processed <= 1 + 1e-6)
    
    # Test with missing values (small gaps)
    data_missing = data.clone()
    data_missing[10:12, 0] = float('nan')  # 2-point gap
    processed = preprocess_data(data_missing, normalize=True, handle_missing=True)
    assert not torch.isnan(processed).any()
    
    # Test without normalization
    processed = preprocess_data(data, normalize=False)
    assert torch.equal(processed, data)
    
    # Test with different accelerators if available
    if HAS_NUMBA:
        processed = preprocess_data(data, normalize=True, use_acceleration=True, accelerator='numba')
        assert processed.shape == data.shape
        assert processed.device == data.device
    
    if HAS_CUPY and torch.cuda.is_available():
        processed = preprocess_data(data, normalize=True, use_acceleration=True, accelerator='cupy')
        assert processed.shape == data.shape
        assert processed.device == data.device

def test_error_handling_comprehensive(preprocessor, device):
    """Test comprehensive error handling in preprocessing."""
    # Test normalize_data errors
    with pytest.raises(ValueError, match="Cannot normalize empty data"):
        preprocessor.normalize_data(torch.tensor([], device=device))
    
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        preprocessor.normalize_data(torch.randn(10, 2, device=device), method='invalid')
    
    # Test handle_missing_values errors
    with pytest.raises(ValueError, match="Unsupported missing value handling method"):
        preprocessor.handle_missing_values(torch.randn(10, 2, device=device), method='invalid')
    
    all_nan_data = torch.full((10, 2), float('nan'), device=device)
    with pytest.raises(ValueError, match="Column 0 contains all NaN values"):
        preprocessor.handle_missing_values(all_nan_data, method='interpolate')
    
    # Test smooth_data errors
    with pytest.raises(ValueError, match="Window size must be positive"):
        preprocessor.smooth_data(torch.randn(10, 2, device=device), window_size=0)
    
    with pytest.raises(ValueError, match="Unsupported smoothing method"):
        preprocessor.smooth_data(torch.randn(10, 2, device=device), method='invalid')
    
    # Test segment_sequences errors
    with pytest.raises(ValueError, match="Segment length must be positive"):
        preprocessor.segment_sequences(torch.randn(10, 2, device=device), segment_length=0)
    
    with pytest.raises(ValueError, match="Overlap must be less than segment length"):
        preprocessor.segment_sequences(torch.randn(10, 2, device=device), segment_length=10, overlap=10)
    
    # Test preprocess_data errors
    with pytest.raises(ValueError, match="Unsupported accelerator"):
        preprocess_data(torch.randn(10, 2, device=device), accelerator='invalid')

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 