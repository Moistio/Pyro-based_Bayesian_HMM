import pytest
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)

@pytest.fixture
def small_data():
    """Create a small dataset for quick tests."""
    return torch.randn(10, 2)

@pytest.fixture
def medium_data():
    """Create a medium-sized dataset for more thorough tests."""
    return torch.randn(100, 2)

@pytest.fixture
def large_data():
    """Create a large dataset for performance tests."""
    return torch.randn(1000, 2)

def pytest_configure(config):
    """Configure pytest to run tests in order."""
    # Disable random ordering
    config.option.random_order = False
    config.option.random_order_seed = 42

def pytest_collection_modifyitems(items):
    """Sort test items to run in order."""
    # Define the order of test functions for each test file
    test_order = [
        # Core model tests
        'test_model_initialization',
        'test_basic_training',
        'test_basic_inference',
        'test_save_load',
        'test_anomaly_detection',
        
        # Preprocessing tests
        'test_handle_missing_values',
        'test_normalize_data',
        'test_standardize_data',
        'test_handle_outliers',
        'test_smooth_data',
        'test_chunk_data',
        'test_batch_processing',
        'test_data_augmentation',
        
        # Metrics tests
        'test_calculate_accuracy',
        'test_calculate_precision',
        'test_calculate_recall',
        'test_calculate_f1',
        'test_calculate_confusion_matrix',
        'test_calculate_roc_auc',
        'test_calculate_log_likelihood',
        'test_calculate_bic',
        'test_calculate_aic'
    ]
    
    def get_test_order(item):
        try:
            return test_order.index(item.name)
        except ValueError:
            # If test name not in list, put it at the end
            return len(test_order)
    
    # Sort items based on the defined order
    items.sort(key=get_test_order) 