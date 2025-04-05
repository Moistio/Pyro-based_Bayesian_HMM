import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, Trace_ELBO, Predictive
from pyro.optim import Adam
from torch.distributions import constraints
from typing import List, Optional, Tuple, Dict, Union
import logging
import numpy as np
from pathlib import Path
import os
import pyro.poutine as poutine
from tqdm.auto import tqdm
from . import preprocessing
from . import metrics
import time

class InferenceNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_states: int, num_layers: int = 1):
        """
        GRU-based inference network for HMM state inference.

        Parameters
        ----------
        input_dim : int
            Dimension of input observations
        hidden_dim : int
            Dimension of hidden GRU layers
        n_states : int
            Number of possible HMM states
        num_layers : int
            Number of GRU layers
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,  
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # bidirectional for richer context
            dropout=0.1 if num_layers > 1 else 0  # Add dropout for regularization
        )
        self.linear = nn.Linear(hidden_dim * 2, n_states)  # *2 for bidirectional
        
    def forward(self, x: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """
        Forward pass of the inference network with batched processing.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, sequence_length, input_dim)
        batch_size : int
            Size of batches for processing
            
        Returns
        -------
        torch.Tensor
            Logits for state probabilities of shape (batch, sequence_length, n_states)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # shape becomes (1, T, input_dim)
            
        # Ensure input is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
            
        # Process in smaller batches to save memory
        seq_len = x.size(1)
        n_batches = (seq_len + batch_size - 1) // batch_size
        outputs = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, seq_len)
            batch = x[:, start_idx:end_idx]
            
            # Process batch
            out, _ = self.gru(batch)
            logits = self.linear(out)
            outputs.append(logits)
            
            # Clear cache between batches
            if i < n_batches - 1:  # Don't clear on last batch
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        return torch.cat(outputs, dim=1)

class PyroBayesianHMM:
    SUPPORTED_EMISSIONS = {
        "gaussian": "Gaussian (Normal) distribution",
        "student_t": "Student's t-distribution",
        "poisson": "Poisson distribution",
        "multinomial": "Multinomial distribution",
        "gamma": "Gamma distribution",
        "beta": "Beta distribution",
    }

    def __init__(
        self, 
        n_states: int, 
        emission_type: str = "gaussian",
        device: str = "cuda",
        obs_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        use_acceleration: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Pyro-based Bayesian Hidden Markov Model with RNN inference.
        """
        self.use_acceleration = use_acceleration
        self.logger = logger or self._setup_logger()
        
        # Set logging level to ERROR to suppress warnings
        logging.getLogger('BayesianHMM').setLevel(logging.ERROR)

        if emission_type not in self.SUPPORTED_EMISSIONS:
            raise ValueError(f"Emission type must be one of: {list(self.SUPPORTED_EMISSIONS.keys())}")
        
        if device == "cuda":
            if not torch.cuda.is_available():
                self.logger.warning("CUDA is not available, falling back to CPU")
                device = "cpu"
            else:
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                self.logger.info(f"CUDA version: {torch.version.cuda}")
                self.logger.info(f"PyTorch version: {torch.__version__}")
        
        # Convert device string to torch.device without index
        self.device = torch.device(device.split(':')[0] if ':' in device else device)

        # Initialize accelerator based on CUDA availability
        self.accelerator = None
        if self.use_acceleration:
            try:
                import cupy as cp
                self.accelerator = "cupy"
                self.logger.info("Using CuPy for acceleration")
            except ImportError:
                try:
                    import numba
                    self.accelerator = "numba"
                    self.logger.info("Using Numba for acceleration")
                except ImportError:
                    self.logger.warning("Neither CuPy nor Numba available, acceleration disabled")
                    self.accelerator = None
        else:
            self.logger.info("Acceleration disabled")

        self.n_states = n_states
        self.emission_type = emission_type
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_trained = False
        self.training_history = []

        # Initialize with more informative priors
        self.initial_probs = torch.ones(n_states, device=self.device) / n_states
        
        # Initialize transition matrix with stronger diagonal (state persistence)
        transition_matrix = torch.ones(n_states, n_states, device=self.device) * 0.1
        for i in range(n_states):
            transition_matrix[i, i] = 1.0  # Stronger self-transition probability
        self.transition_probs = transition_matrix / transition_matrix.sum(dim=1, keepdim=True)
        
        if emission_type == "gaussian":
            # Initialize with more spread out means for better state separation
            means = torch.linspace(-2, 2, n_states, device=self.device).unsqueeze(1).repeat(1, obs_dim)
            self.emission_params = {
                "loc": means,
                "scale": torch.ones(n_states, obs_dim, device=self.device)
            }
        elif emission_type == "poisson":
            # Initialize with different rates for each state
            rates = torch.linspace(0.5, 2.0, n_states, device=self.device).unsqueeze(1).repeat(1, obs_dim)
            self.emission_params = {
                "rate": rates
            }
        else:  # multinomial
            self.emission_params = {
                "probs": torch.ones(n_states, obs_dim, device=self.device) / obs_dim
            }

        # Initialize inference network
        self.inference_net = InferenceNetwork(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            n_states=n_states,
            num_layers=num_layers
        ).to(self.device)
        pyro.module("inference_net", self.inference_net)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("BayesianHMM")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    @config_enumerate
    def model(self, data: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Define the model."""
        # Define initial state probabilities
        initial_probs_param = pyro.param(
            "initial_probs_param",
            torch.ones(self.n_states, device=self.device) / self.n_states,
            constraint=dist.constraints.simplex
        )
        initial_probs = pyro.sample("initial_probs", dist.Dirichlet(initial_probs_param))
        
        # Define transition probabilities
        transition_probs_param = pyro.param(
            "transition_probs_param",
            torch.ones(self.n_states, self.n_states, device=self.device) / self.n_states,
            constraint=dist.constraints.simplex
        )
        transition_probs = torch.zeros_like(transition_probs_param)
        for i in range(self.n_states):
            transition_probs[i] = pyro.sample(
                f"transition_probs_{i}",
                dist.Dirichlet(transition_probs_param[i])
            )
        
        # Emission parameters
        if self.emission_type == "gaussian":
            locs = []
            scales = []
            for i in range(self.n_states):
                loc_param = pyro.param(f"loc_param_{i}", torch.zeros(self.obs_dim, device=self.device))
                scale_param = pyro.param(f"scale_param_{i}", torch.ones(self.obs_dim, device=self.device), constraint=dist.constraints.positive)
                loc = pyro.sample(f"loc_{i}", dist.Normal(loc_param, 0.1).to_event(1))
                scale = pyro.sample(f"scale_{i}", dist.Gamma(scale_param, 1.0).to_event(1))
                locs.append(loc)
                scales.append(scale)
            locs = torch.stack(locs)
            scales = torch.stack(scales)
        elif self.emission_type == "poisson":
            rates = []
            for i in range(self.n_states):
                rate_param = pyro.param(f"rate_param_{i}", torch.ones(self.obs_dim, device=self.device), constraint=dist.constraints.positive)
                rate = pyro.sample(f"rate_{i}", dist.Gamma(rate_param, 1.0).to_event(1))
                rates.append(rate)
            rates = torch.stack(rates)
        else:
            probs_list = []
            for i in range(self.n_states):
                probs_param = pyro.param(
                    f"probs_param_{i}",
                    torch.ones(self.obs_dim, device=self.device) / self.obs_dim,
                    constraint=dist.constraints.simplex
                )
                probs = pyro.sample(f"probs_{i}", dist.Dirichlet(probs_param).to_event(1))
                probs_list.append(probs)
            probs_list = torch.stack(probs_list)
        
        # Sample states and observations
        states = []
        for t in range(data.shape[0]):
            if t == 0:
                state = pyro.sample(f"state_{t}", dist.Categorical(initial_probs))
            else:
                prev_state = states[-1]
                state = pyro.sample(
                    f"state_{t}",
                    dist.Categorical(transition_probs[prev_state])
                )
            states.append(state)
            
            if mask is None or not mask[t].any():
                curr_state = state
                if self.emission_type == "gaussian":
                    curr_loc = locs[curr_state].expand(data[t].shape)
                    curr_scale = scales[curr_state].expand(data[t].shape)
                    pyro.sample(f"obs_{t}", dist.Normal(curr_loc, curr_scale).to_event(1), obs=data[t])
                elif self.emission_type == "poisson":
                    curr_rate = rates[curr_state].expand(data[t].shape)
                    pyro.sample(f"obs_{t}", dist.Poisson(curr_rate).to_event(1), obs=data[t])
                else:
                    curr_probs = probs_list[curr_state].expand(data[t].shape)
                    pyro.sample(f"obs_{t}", dist.Multinomial(1, curr_probs).to_event(1), obs=data[t])

    def guide(self, data: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Define the guide."""
        # Add batch dimension if needed
        if data.dim() == 2:
            data = data.unsqueeze(0)  # shape becomes (1, T, obs_dim)
        
        # Get logits from the inference network
        logits = self.inference_net(data)  # shape: (batch, T, n_states)
        logits = logits.squeeze(0)  # remove batch dimension, now (T, n_states)
        
        # Sample states using the network's logits
        states = []
        for t in range(data.shape[1]):
            if mask is not None and mask[t].any():
                # For masked positions, use uniform distribution
                state = pyro.sample(
                    f"state_{t}",
                    dist.Categorical(torch.ones(self.n_states, device=self.device) / self.n_states)
                )
            else:
                # Use network's logits for unmasked positions
                state = pyro.sample(
                    f"state_{t}",
                    dist.Categorical(logits=logits[t])
                )
            states.append(state)
        
        # Sample other parameters
        initial_probs_guide = pyro.param(
            "initial_probs_guide",
            torch.ones(self.n_states, device=self.device) / self.n_states,
            constraint=dist.constraints.simplex
        )
        initial_probs = pyro.sample("initial_probs", dist.Dirichlet(initial_probs_guide))
        
        transition_probs_guide = pyro.param(
            "transition_probs_guide",
            torch.ones(self.n_states, self.n_states, device=self.device) / self.n_states,
            constraint=dist.constraints.simplex
        )
        transition_probs = torch.zeros_like(transition_probs_guide)
        for i in range(self.n_states):
            transition_probs[i] = pyro.sample(
                f"transition_probs_{i}",
                dist.Dirichlet(transition_probs_guide[i])
            )
        
        if self.emission_type == "gaussian":
            for i in range(self.n_states):
                loc_guide = pyro.param(f"loc_guide_{i}", torch.zeros(self.obs_dim, device=self.device))
                scale_guide = pyro.param(f"scale_guide_{i}", torch.ones(self.obs_dim, device=self.device), constraint=dist.constraints.positive)
                pyro.sample(f"loc_{i}", dist.Normal(loc_guide, 0.1).to_event(1))
                pyro.sample(f"scale_{i}", dist.Gamma(scale_guide, 1.0).to_event(1))
        elif self.emission_type == "poisson":
            for i in range(self.n_states):
                rate_guide = pyro.param(f"rate_guide_{i}", torch.ones(self.obs_dim, device=self.device), constraint=dist.constraints.positive)
                pyro.sample(f"rate_{i}", dist.Gamma(rate_guide, 1.0).to_event(1))
        else:  # multinomial
            for i in range(self.n_states):
                probs_guide = pyro.param(
                    f"probs_guide_{i}",
                    torch.ones(self.obs_dim, device=self.device) / self.obs_dim,
                    constraint=dist.constraints.simplex
                )
                pyro.sample(f"probs_{i}", dist.Dirichlet(probs_guide).to_event(1))

    def train(
        self,
        data: torch.Tensor,
        num_steps: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        mask: Optional[torch.Tensor] = None
    ) -> List[float]:
        """Train the model with dynamically set hyperparameters."""
        self.logger.info("Starting training...")
        
        # Ensure data is on the correct device
        data = data.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
            
        # Preprocess data
        self.data = preprocessing.preprocess_data(
            data, 
            normalize=True, 
            use_acceleration=self.use_acceleration, 
            accelerator=self.accelerator
        ).to(self.device)
        
        # Dynamically set hyperparameters based on data characteristics
        n_samples = self.data.shape[0]
        n_features = self.data.shape[1]
        
        # Calculate optimal batch size based on available GPU memory
        if batch_size is None:
            if torch.cuda.is_available():
                # Get available GPU memory in GB
                available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Estimate memory per sample (conservative estimate)
                memory_per_sample = (n_features * 4) / (1024**3)  # 4 bytes per float
                # Calculate maximum batch size based on available memory
                max_batch_size = int(available_memory * 0.3 / memory_per_sample)  # Use 30% of available memory
                batch_size = min(64, max(16, max_batch_size))
                self.logger.info(f"GPU memory: {available_memory:.2f}GB, Estimated memory per sample: {memory_per_sample:.4f}GB")
            else:
                # CPU fallback
                batch_size = min(32, max(8, n_samples // 100))
            self.logger.info(f"Automatically set batch size to {batch_size}")
            
        # Set learning rate based on data characteristics
        if learning_rate is None:
            # Use smaller learning rate for larger datasets
            base_lr = 0.0001
            scale_factor = min(1.0, 1000 / n_samples)
            learning_rate = base_lr * scale_factor
            self.logger.info(f"Automatically set learning rate to {learning_rate:.6f}")
            
        # Set number of steps based on data size and complexity
        if num_steps is None:
            # More steps for larger datasets and more states
            base_steps = 50
            complexity_factor = max(1.0, n_samples / 1000) * (self.n_states / 3)
            num_steps = int(base_steps * complexity_factor)
            self.logger.info(f"Automatically set number of steps to {num_steps}")
            
        # Set gradient clipping threshold based on data scale
        data_std = torch.std(self.data).item()
        max_grad_norm = min(0.1, max(0.01, data_std / 10))
        self.logger.info(f"Automatically set gradient clipping to {max_grad_norm:.4f}")
            
        optimizer = Adam({"lr": learning_rate})
        svi = SVI(self.model, self.guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))
        losses = []
        best_loss = float('inf')
        patience = max(5, min(20, num_steps // 10))  # Dynamic patience
        patience_counter = 0
        current_lr = learning_rate
        min_lr = learning_rate * 0.01
        
        # Create progress bar
        pbar = tqdm(range(num_steps), desc="Training", unit="step", leave=True)
        
        for step in pbar:
            try:
                if batch_size is not None:
                    # Use stratified sampling for better training
                    indices = torch.randperm(self.data.shape[0])[:batch_size]
                    batch_data = self.data[indices]
                    batch_mask = mask[indices] if mask is not None else None
                    
                    # Clear cache before processing batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    loss = svi.step(batch_data, batch_mask)
                else:
                    loss = svi.step(self.data, mask)
                    
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(pyro.get_param_store().values(), max_grad_norm)
                
                # Update best loss and patience
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Learning rate scheduling with warmup
                if step < 5:  # Warmup period
                    current_lr = learning_rate * (step + 1) / 5
                    optimizer = Adam({"lr": current_lr})
                    svi = SVI(self.model, self.guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))
                elif patience_counter >= patience:
                    current_lr = max(current_lr * 0.5, min_lr)
                    optimizer = Adam({"lr": current_lr})
                    svi = SVI(self.model, self.guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))
                    patience_counter = 0
                    self.logger.info(f"Reducing learning rate to {current_lr:.6f}")
                
                losses.append(loss)
                
                # Update progress bar with more detailed information
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{current_lr:.6f}',
                    'patience': patience_counter,
                    'best_loss': f'{best_loss:.4f}'
                })
                
                # Log every 5 steps
                if step % 5 == 0:
                    self.logger.info(f"Step {step}: loss = {loss:.4f}, lr = {current_lr:.6f}")
                
                # Early stopping
                if current_lr <= min_lr and patience_counter >= patience:
                    self.logger.info("Early stopping triggered")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error during training step {step}: {str(e)}")
                raise
                
        self.model_trained = True
        self.training_history = losses
        self.logger.info("Training completed successfully")
        return losses

    def _calculate_optimal_samples(self, data: torch.Tensor) -> int:
        """Calculate optimal number of samples for inference based on data characteristics.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data tensor

        Returns
        -------
        int
            Optimal number of samples for inference
        """
        n_samples, n_features = data.shape
        
        # Base number of samples
        base_samples = 30
        
        # Adjust based on data size
        size_factor = min(1.0, 1000 / n_samples)  # Fewer samples for larger datasets
        samples = int(base_samples * size_factor)
        
        # Adjust based on number of states
        state_factor = min(1.0, 3 / self.n_states)  # Fewer samples for more states
        samples = int(samples * state_factor)
        
        # Adjust based on available GPU memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = torch.cuda.memory_allocated(0)
            available_memory = total_memory - free_memory
            
            # Estimate memory per sample (bytes)
            memory_per_sample = n_features * 4 * self.n_states  # 4 bytes per float32
            memory_overhead = 2.0  # Factor for PyTorch overhead
            
            # Calculate maximum samples based on available memory
            max_samples = int((available_memory * 0.3) / (memory_per_sample * memory_overhead))
            samples = min(samples, max_samples)
        
        # Ensure minimum and maximum bounds
        samples = max(20, min(100, samples))
        
        return samples

    def infer_states(self, data: torch.Tensor, num_samples: Optional[int] = None) -> torch.Tensor:
        """Infer the most likely states for the given data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data tensor
        num_samples : Optional[int]
            Number of samples to use for inference. If None, will be calculated automatically.
            
        Returns
        -------
        torch.Tensor
            Most likely state for each timestep
        """
        if not self.model_trained:
            raise ValueError("Model must be trained before inference")
        
        start_time = time.time()
        
        # Calculate optimal number of samples if not provided
        if num_samples is None:
            num_samples = self._calculate_optimal_samples(data)
            self.logger.info(f"Automatically set number of samples to {num_samples}")
        
        self.logger.info(f"Starting state inference with {num_samples} samples...")
        self.logger.info(f"Input data shape: {data.shape}")
        
        # Process in smaller batches to save memory
        batch_size = min(500, data.shape[0])  # Reduced batch size
        n_batches = (data.shape[0] + batch_size - 1) // batch_size
        all_states = []
        
        try:
            with tqdm(total=n_batches, desc="Processing batches", unit="batch") as pbar:
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, data.shape[0])
                    batch_data = data[start_idx:end_idx]
                    
                    # Clear cache before processing batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    try:
                        # Generate samples for this batch with error handling
                        batch_posterior = Predictive(
                            self.model,
                            guide=self.guide,
                            num_samples=num_samples,
                            return_sites=("state_.*",)  # Only return state variables
                        )(batch_data)
                        
                        # Extract states for this batch with validation
                        batch_states = []
                        for t in range(end_idx - start_idx):
                            state_key = f"state_{t}"
                            if state_key not in batch_posterior:
                                self.logger.warning(f"Missing state for timestep {t}, using default state 0")
                                batch_states.append(torch.tensor(0, device=self.device))
                                continue
                                
                            state_samples = batch_posterior[state_key]
                            # Ensure states are within valid range
                            valid_samples = torch.clamp(state_samples, 0, self.n_states - 1)
                            # Get most common state
                            state = torch.mode(valid_samples, dim=0)[0]
                            batch_states.append(state)
                        
                        # Stack batch states
                        batch_states = torch.stack(batch_states)
                        all_states.append(batch_states)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing batch {i}: {str(e)}")
                        # Use previous states or default state for this batch
                        default_states = torch.zeros(end_idx - start_idx, device=self.device)
                        all_states.append(default_states)
                    
                    pbar.update(1)
                    
                    # Log progress periodically
                    if (i + 1) % max(1, n_batches // 10) == 0:
                        self.logger.info(f"Processed {i + 1}/{n_batches} batches")
        
        except Exception as e:
            self.logger.error(f"Error during state inference: {str(e)}")
            # Return default states in case of error
            return torch.zeros(data.shape[0], device=self.device)
        
        # Combine all states and ensure they're on the correct device
        try:
            states = torch.cat(all_states, dim=0).to(self.device)
            # Final validation of states
            states = torch.clamp(states, 0, self.n_states - 1)
        except Exception as e:
            self.logger.error(f"Error combining states: {str(e)}")
            return torch.zeros(data.shape[0], device=self.device)
        
        total_time = time.time() - start_time
        self.logger.info(f"Inference completed in {total_time:.2f} seconds")
        self.logger.info(f"Output shape: {states.shape}")
        
        # Log state distribution
        unique_states, counts = torch.unique(states, return_counts=True)
        state_dist = {int(s): int(c) for s, c in zip(unique_states.cpu(), counts.cpu())}
        self.logger.info(f"State distribution: {state_dist}")
        
        return states

    def infer_log_prob(self, data: torch.Tensor) -> float:
        if not self.model_trained:
            raise ValueError("Model must be trained before inference")
            
        start_time = time.time()
        self.logger.info("Starting log probability inference...")
        self.logger.info(f"Input data shape: {data.shape}")
        
        # Calculate optimal number of samples
        num_samples = self._calculate_optimal_samples(data)
        self.logger.info(f"Using {num_samples} samples for log probability inference")
        
        data = data.to(self.device)
        with torch.no_grad():
            # Generate samples
            sample_start = time.time()
            self.logger.info("Generating predictive samples...")
            predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
            samples = predictive(data)
            sample_time = time.time() - sample_start
            self.logger.info(f"Sample generation completed in {sample_time:.2f} seconds")
            
            # Compute log probabilities
            log_start = time.time()
            self.logger.info("Computing log probabilities...")
            log_probs = []
            total_timesteps = data.shape[0]
            
            # Process in batches for better memory management
            batch_size = min(100, total_timesteps)
            n_batches = (total_timesteps + batch_size - 1) // batch_size
            
            with tqdm(total=total_timesteps, desc="Computing log probabilities", unit="timestep") as pbar:
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, total_timesteps)
                    
                    for t in range(start_idx, end_idx):
                        # Get the state for this timestep
                        state_samples = samples[f"state_{t}"]  # shape: [num_samples]
                        
                        # Initialize batch log probs
                        batch_log_probs = torch.zeros(num_samples, device=self.device)
                        
                        # Compute log prob for each sample
                        for s in range(num_samples):
                            state = state_samples[s].item()  # Get scalar state value
                            
                            # Get the emission parameters for this state
                            if self.emission_type == "gaussian":
                                loc = samples[f"loc_{state}"]  # shape: [num_samples, obs_dim]
                                scale = samples[f"scale_{state}"]  # shape: [num_samples, obs_dim]
                                # Create a multivariate normal distribution for each sample
                                emission_dist = pyro.distributions.Normal(
                                    loc[s],  # shape: [obs_dim]
                                    scale[s]  # shape: [obs_dim]
                                )
                                # Compute log probability for each dimension and sum
                                batch_log_probs[s] = emission_dist.log_prob(data[t]).sum()
                            elif self.emission_type == "poisson":
                                rate = samples[f"rate_{state}"]  # shape: [num_samples, obs_dim]
                                emission_dist = pyro.distributions.Poisson(
                                    rate[s]  # shape: [obs_dim]
                                )
                                # Compute log probability for each dimension and sum
                                batch_log_probs[s] = emission_dist.log_prob(data[t]).sum()
                            else:  # multinomial
                                probs = samples[f"probs_{state}"]  # shape: [num_samples, obs_dim]
                                emission_dist = pyro.distributions.Multinomial(
                                    1, probs[s]  # shape: [obs_dim]
                                )
                                # Compute log probability for each dimension and sum
                                batch_log_probs[s] = emission_dist.log_prob(data[t]).sum()
                        
                        # Average over samples
                        log_probs.append(batch_log_probs.mean())
                        pbar.update(1)
                    
                    # Clear cache between batches
                    if torch.cuda.is_available() and i < n_batches - 1:
                        torch.cuda.empty_cache()
            
            log_time = time.time() - log_start
            self.logger.info(f"Log probability computation completed in {log_time:.2f} seconds")
            
            # Final computation
            final_start = time.time()
            self.logger.info("Computing final log probability...")
            log_prob = torch.stack(log_probs).sum()
            final_time = time.time() - final_start
            self.logger.info(f"Final computation completed in {final_time:.2f} seconds")
            
            total_time = time.time() - start_time
            self.logger.info(f"Total inference completed in {total_time:.2f} seconds")
            self.logger.info(f"Log probability computed: {log_prob.item():.4f}")
        return log_prob.item()

    def detect_anomalies(self, data: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        if not self.model_trained:
            raise ValueError("Model must be trained before anomaly detection")
            
        start_time = time.time()
        self.logger.info("Starting anomaly detection...")
        self.logger.info(f"Input data shape: {data.shape}")
        
        # Calculate optimal number of samples
        num_samples = self._calculate_optimal_samples(data)
        self.logger.info(f"Using {num_samples} samples for anomaly detection")
        
        # Ensure data is on the correct device without index
        data = data.to(self.device.type)
        with torch.no_grad():
            # Generate samples
            sample_start = time.time()
            self.logger.info("Generating predictive samples...")
            predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
            samples = predictive(data)
            sample_time = time.time() - sample_start
            self.logger.info(f"Sample generation completed in {sample_time:.2f} seconds")
            
            # Compute log probabilities
            log_start = time.time()
            self.logger.info("Computing log probabilities for anomaly detection...")
            log_probs = []
            total_timesteps = data.shape[0]
            
            # Process in batches for better memory management
            batch_size = min(100, total_timesteps)
            n_batches = (total_timesteps + batch_size - 1) // batch_size
            
            with tqdm(total=total_timesteps, desc="Computing log probabilities", unit="timestep") as pbar:
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, total_timesteps)
                    
                    for t in range(start_idx, end_idx):
                        # Get the state for this timestep
                        state_samples = samples[f"state_{t}"]  # shape: [num_samples]
                        
                        # Initialize batch log probs
                        batch_log_probs = torch.zeros(num_samples, device=self.device.type)
                        
                        # Compute log prob for each sample
                        for s in range(num_samples):
                            state = state_samples[s].item()  # Get scalar state value
                            
                            # Get the emission parameters for this state
                            if self.emission_type == "gaussian":
                                loc = samples[f"loc_{state}"]  # shape: [num_samples, obs_dim]
                                scale = samples[f"scale_{state}"]  # shape: [num_samples, obs_dim]
                                # Create a multivariate normal distribution for each sample
                                emission_dist = pyro.distributions.Normal(
                                    loc[s],  # shape: [obs_dim]
                                    scale[s]  # shape: [obs_dim]
                                )
                                # Compute log probability for each dimension and sum
                                batch_log_probs[s] = emission_dist.log_prob(data[t]).sum()
                            elif self.emission_type == "poisson":
                                rate = samples[f"rate_{state}"]  # shape: [num_samples, obs_dim]
                                emission_dist = pyro.distributions.Poisson(
                                    rate[s]  # shape: [obs_dim]
                                )
                                # Compute log probability for each dimension and sum
                                batch_log_probs[s] = emission_dist.log_prob(data[t]).sum()
                            else:  # multinomial
                                probs = samples[f"probs_{state}"]  # shape: [num_samples, obs_dim]
                                emission_dist = pyro.distributions.Multinomial(
                                    1, probs[s]  # shape: [obs_dim]
                                )
                                # Compute log probability for each dimension and sum
                                batch_log_probs[s] = emission_dist.log_prob(data[t]).sum()
                        
                        # Store the mean log probability for this timestep
                        log_probs.append(batch_log_probs.mean())
                        pbar.update(1)
                    
                    # Clear cache between batches
                    if torch.cuda.is_available() and i < n_batches - 1:
                        torch.cuda.empty_cache()
            
            log_time = time.time() - log_start
            self.logger.info(f"Log probability computation completed in {log_time:.2f} seconds")
            
            # Convert log probabilities to tensor
            log_probs = torch.tensor(log_probs, device=self.device.type)
            
            # Compute mean and standard deviation
            mean_log_prob = log_probs.mean()
            std_log_prob = log_probs.std()
            
            # Detect anomalies
            anomalies = torch.abs(log_probs - mean_log_prob) > threshold * std_log_prob
            
            total_time = time.time() - start_time
            self.logger.info(f"Anomaly detection completed in {total_time:.2f} seconds")
            self.logger.info(f"Found {anomalies.sum().item()} anomalies out of {len(anomalies)} samples")
            
        return anomalies

    @classmethod
    def load_model(cls, path: str, device: str = "cuda") -> "PyroBayesianHMM":
        params = torch.load(path, map_location=device)
        model = cls(n_states=params["n_states"], 
                   emission_type=params["emission_type"],
                   device=device,
                   obs_dim=params["obs_dim"])
        for name, param in params.items():
            if name not in ["n_states", "emission_type", "obs_dim"]:
                pyro.param(name, param.to(model.device))
        model.model_trained = True
        return model

    def save_model(self, path: str):
        params = {
            "n_states": self.n_states,
            "emission_type": self.emission_type,
            "obs_dim": self.obs_dim
        }
        for name, param in pyro.get_param_store().items():
            params[name] = param.detach().cpu()
        torch.save(params, path)

    def plot_training_history(self):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(self.training_history)
            plt.title('Training Loss History')
            plt.xlabel('Step')
            plt.ylabel('ELBO Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.show()
        except ImportError:
            self.logger.warning("matplotlib is required for plotting")

    def _stratified_sample(self, data: torch.Tensor, batch_size: int, states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform stratified sampling to ensure balanced batches.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data tensor
        batch_size : int
            Desired batch size
        states : Optional[torch.Tensor]
            Current state assignments if available
            
        Returns
        -------
        torch.Tensor
            Indices for stratified batch
        """
        if states is None:
            # If no states provided, use current model to infer them
            if self.model_trained:
                with torch.no_grad():
                    states = self.infer_states(data)
            else:
                # For initial epochs, use random assignments
                states = torch.randint(0, self.n_states, (len(data),), device=self.device)
        
        # Calculate samples per state
        samples_per_state = batch_size // self.n_states
        remaining = batch_size % self.n_states
        
        indices = []
        for state in range(self.n_states):
            state_indices = torch.where(states == state)[0]
            if len(state_indices) > 0:
                # Sample with replacement if we need more samples than available
                num_samples = samples_per_state + (1 if state < remaining else 0)
                if len(state_indices) < num_samples:
                    sampled = state_indices[torch.randint(len(state_indices), (num_samples,))]
                else:
                    sampled = state_indices[torch.randperm(len(state_indices))[:num_samples]]
                indices.append(sampled)
        
        # Combine and shuffle
        indices = torch.cat(indices)
        indices = indices[torch.randperm(len(indices))]
        return indices

    def _calculate_optimal_params(self, data: torch.Tensor) -> dict:
        """Calculate optimal training parameters based on data characteristics.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data tensor

        Returns
        -------
        dict
            Dictionary containing optimal parameters
        """
        n_samples, n_features = data.shape
        
        # Calculate optimal chunk size based on data size and available memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = torch.cuda.memory_allocated(0)
            available_memory = total_memory - free_memory
            
            # Estimate memory per sample (bytes)
            memory_per_sample = n_features * 4  # 4 bytes per float32
            memory_overhead = 1.5  # Factor for PyTorch overhead
            
            # Target using 20% of available memory for chunks
            optimal_chunk_size = int((available_memory * 0.2) / (memory_per_sample * memory_overhead))
            # Ensure chunk size is reasonable
            chunk_size = min(max(100, optimal_chunk_size), 1000)
        else:
            # CPU fallback: use smaller chunks
            chunk_size = min(max(100, n_samples // 20), 500)
        
        # Calculate optimal number of steps based on data complexity
        complexity_factor = (n_features * self.n_states) / 10
        base_steps = 50
        num_steps = int(base_steps * complexity_factor)
        num_steps = min(max(50, num_steps), 200)  # Keep reasonable bounds
        
        # Calculate optimal batch chunks based on available memory
        if torch.cuda.is_available():
            optimal_batch_chunks = int((available_memory * 0.1) / (chunk_size * memory_per_sample * memory_overhead))
            batch_chunks = min(max(4, optimal_batch_chunks), 16)
        else:
            batch_chunks = 4
        
        # Calculate learning rate based on data scale
        data_scale = torch.std(data).item()
        base_lr = 0.001
        learning_rate = base_lr * min(1.0, data_scale)
        
        # Calculate patience based on number of steps
        patience = max(5, num_steps // 10)
        
        # Calculate minimum delta based on data scale
        min_delta = max(1e-5, data_scale * 0.001)
        
        return {
            'chunk_size': chunk_size,
            'num_steps': num_steps,
            'batch_chunks': batch_chunks,
            'learning_rate': learning_rate,
            'patience': patience,
            'min_delta': min_delta
        }

    def train_chunked(
        self,
        data: torch.Tensor,
        chunk_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        learning_rate: Optional[float] = None,
        patience: Optional[int] = None,
        min_delta: Optional[float] = None,
        batch_chunks: Optional[int] = None,
        use_stratified: bool = True,
        target_memory_gb: float = 1.0
    ) -> List[float]:
        """Train the model using chunked strategy with dynamic parameters."""
        self.logger.info("Starting chunked training...")
        
        # Calculate optimal parameters if not provided
        optimal_params = self._calculate_optimal_params(data)
        chunk_size = chunk_size or optimal_params['chunk_size']
        num_steps = num_steps or optimal_params['num_steps']
        learning_rate = learning_rate or optimal_params['learning_rate']
        patience = patience or optimal_params['patience']
        min_delta = min_delta or optimal_params['min_delta']
        batch_chunks = batch_chunks or optimal_params['batch_chunks']
        
        # Log the parameters being used
        self.logger.info("Training parameters:")
        self.logger.info(f"Chunk size: {chunk_size}")
        self.logger.info(f"Number of steps: {num_steps}")
        self.logger.info(f"Learning rate: {learning_rate:.6f}")
        self.logger.info(f"Patience: {patience}")
        self.logger.info(f"Min delta: {min_delta:.6f}")
        self.logger.info(f"Batch chunks: {batch_chunks}")
        
        # Preprocess data using acceleration
        data = preprocessing.preprocess_data(data, normalize=True, use_acceleration=self.use_acceleration, accelerator=self.accelerator)
        
        # Get initial state assignments if using stratified sampling
        initial_states = None
        if use_stratified:
            self.logger.info("Using stratified sampling...")
            if not self.model_trained:
                initial_states = torch.randint(0, self.n_states, (data.shape[0],), device=self.device)
            else:
                initial_states = self.infer_states(data)
            
            chunks = preprocessing.chunk_data(data, chunk_size, use_acceleration=self.use_acceleration, accelerator=self.accelerator)
        else:
            chunks = preprocessing.chunk_data(data, chunk_size, use_acceleration=self.use_acceleration, accelerator=self.accelerator)
            
        self.logger.info(f"Created {len(chunks)} chunks for training")
        
        optimizer = Adam({"lr": learning_rate})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        losses = []
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        
        # Learning rate scheduling
        min_lr = learning_rate * 0.01
        current_lr = learning_rate
        
        # Add progress bar with more detailed information
        pbar = tqdm(range(num_steps), desc="Training", unit="step", position=0, leave=True)
        
        # Add chunk progress bar
        chunk_pbar = tqdm(total=batch_chunks, desc="Processing chunks", unit="chunk", position=1, leave=False)
        
        for step in pbar:
            try:
                # Randomly select chunks for this step
                chunk_indices = torch.randperm(len(chunks))[:batch_chunks]
                step_loss = 0.0
                
                # Reset chunk progress bar
                chunk_pbar.reset()
                
                # Process selected chunks
                for idx in chunk_indices:
                    chunk_data = chunks[idx].to(self.device)
                    
                    if use_stratified:
                        # Get stratified sample indices
                        batch_indices = self._stratified_sample(
                            chunk_data, 
                            min(chunk_size, len(chunk_data)),
                            states=initial_states[idx*chunk_size:(idx+1)*chunk_size] if initial_states is not None else None
                        )
                        chunk_data = chunk_data[batch_indices]
                    
                    # Clear CUDA cache before processing chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    loss = svi.step(chunk_data)
                    step_loss += loss
                    
                    # Update chunk progress
                    chunk_pbar.update(1)
                    chunk_pbar.set_postfix({'chunk_loss': f'{loss:.4f}'})
                
                # Average loss for this step
                avg_loss = step_loss / batch_chunks
                losses.append(avg_loss)
                
                # Early stopping check
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Learning rate scheduling
                if patience_counter >= patience:
                    current_lr = max(current_lr * 0.5, min_lr)
                    optimizer = Adam({"lr": current_lr})
                    svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
                    patience_counter = 0
                    self.logger.info(f"Reducing learning rate to {current_lr:.6f}")
                
                # Update main progress bar with more information
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.6f}',
                    'patience': patience_counter,
                    'best_loss': f'{best_loss:.4f}'
                })
                
                # Early stopping
                if current_lr <= min_lr and patience_counter >= patience:
                    self.logger.info("Early stopping triggered")
                    break
            except Exception as e:
                self.logger.error(f"Error during training step {step}: {str(e)}")
                raise
            
            # Add a small delay to prevent CPU overload
            if step % 10 == 0:
                torch.cuda.synchronize()
        
        # Close progress bars
        pbar.close()
        chunk_pbar.close()
        
        self.model_trained = True
        self.training_history = losses
        self.logger.info("Chunked training completed successfully")
        return losses
