# Pyro-based Bayesian Hidden Markov Model

A GPU-accelerated implementation of a Bayesian Hidden Markov Model using Pyro for anomaly detection in time series data. This project combines probabilistic programming with deep learning techniques to provide robust anomaly detection capabilities.

## Features

- **Bayesian HMM Implementation**
  - Probabilistic state inference
  - Automatic parameter optimization
  - Dynamic learning rate scheduling
  - Stratified sampling support

- **Performance Optimizations**
  - GPU acceleration
  - Chunked data processing
  - Memory-efficient operations
  - Batch processing capabilities

## Project Structure

```
.
├── src/                      # Source code
│   ├── __init__.py
│   ├── pyroBayessianHMM.py  # Core HMM implementation
│   ├── preprocessing.py      # Data preprocessing utilities
│   └── metrics.py           # Evaluation metrics
├── tests/                    # Test suite
│   ├── test_pyro_bayesian_hmm.py
│   ├── test_preprocessing.py
│   └── test_metrics.py
├── notebooks/               # Examples and demos
│   └── demo.py             # Main demo script
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic example:

```python
from src.pyroBayessianHMM import PyroBayesianHMM

# Initialize model
model = PyroBayesianHMM(
    n_states=3,
    emission_type="gaussian",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train model
losses = model.train_chunked(data)

# Detect anomalies
anomalies = model.detect_anomalies(test_data, threshold=1.5)
```

For a complete example with synthetic data generation and visualization, run:
```bash
python notebooks/demo.py
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Generate coverage report:
```bash
pytest --cov=src tests/
```

## Model Architecture

The model uses:
- Bayesian inference with Pyro
- GRU-based inference network
- Automatic parameter optimization
- Dynamic learning rate scheduling
- GPU acceleration when available

## Performance

On the demo dataset (2000 samples, 1.5% anomaly rate):
- Precision: ~0.7
- Recall: ~0.8
- F1 Score: ~0.75

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License

## References

- Pyro: Deep Universal Probabilistic Programming (https://pyro.ai/)
- "Structured Inference Networks for Nonlinear State Space Models" (Johnson et al., 2016) 