"""
RAPTOR: Real-time Adaptive Privacy Through Output Regulation

Main framework class that integrates RPLQ, AOS, and PUPO into a unified
privacy protection system for ML models at deployment time.

This implements the closed-loop architecture described in the paper,
where privacy measurement informs adaptive defense through online optimization.

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
from functools import wraps
import logging
import time
import yaml

from .rplq import RPLQ, RPLQConfig
from ..defenses.aos import AOS, AOSConfig
from ..optimization.pupo import PUPO, PUPOConfig

logger = logging.getLogger(__name__)


@dataclass
class RAPTORConfig:
    """Master configuration for RAPTOR framework."""
    # Component configs
    rplq: RPLQConfig = None
    aos: AOSConfig = None
    pupo: PUPOConfig = None
    
    # Framework settings
    enable_feedback_loop: bool = True
    log_queries: bool = True
    latency_budget_ms: float = 10.0
    
    # Attack simulation for PUPO training
    simulate_attacks: bool = True
    attack_sample_rate: float = 0.1
    
    def __post_init__(self):
        if self.rplq is None:
            self.rplq = RPLQConfig()
        if self.aos is None:
            self.aos = AOSConfig()
        if self.pupo is None:
            self.pupo = PUPOConfig()
    
    @classmethod
    def from_yaml(cls, path: str) -> 'RAPTORConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        rplq_config = RPLQConfig(**config_dict.get('rplq', {}))
        aos_config = AOSConfig(**config_dict.get('aos', {}))
        pupo_config = PUPOConfig(**config_dict.get('pupo', {}))
        
        return cls(
            rplq=rplq_config,
            aos=aos_config,
            pupo=pupo_config,
            **{k: v for k, v in config_dict.items() if k not in ['rplq', 'aos', 'pupo']}
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'rplq': self.rplq.__dict__,
            'aos': self.aos.__dict__,
            'pupo': self.pupo.__dict__,
            'enable_feedback_loop': self.enable_feedback_loop,
            'log_queries': self.log_queries,
            'latency_budget_ms': self.latency_budget_ms,
            'simulate_attacks': self.simulate_attacks,
            'attack_sample_rate': self.attack_sample_rate
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class QueryLogger:
    """Logs queries and their privacy metrics for analysis."""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.entries = []
    
    def log(
        self,
        query_id: int,
        risk_scores: Dict[str, float],
        sigma: float,
        latency_ms: float
    ):
        """Log a query."""
        entry = {
            'query_id': query_id,
            'timestamp': time.time(),
            **risk_scores,
            'sigma': sigma,
            'latency_ms': latency_ms
        }
        
        self.entries.append(entry)
        
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get aggregate statistics."""
        if len(self.entries) == 0:
            return {}
        
        risks = [e['rho'] for e in self.entries]
        latencies = [e['latency_ms'] for e in self.entries]
        
        return {
            'num_queries': len(self.entries),
            'mean_risk': np.mean(risks),
            'max_risk': np.max(risks),
            'mean_latency_ms': np.mean(latencies),
            'p99_latency_ms': np.percentile(latencies, 99)
        }


class AttackSimulator:
    """
    Simulates privacy attacks for PUPO feedback.
    
    In deployment, this would be replaced with actual attack monitoring.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def simulate_mia(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        sanitized_output: torch.Tensor
    ) -> float:
        """
        Simulate membership inference attack success.
        
        Returns attack success rate based on output difference.
        """
        # Simplified simulation: attack success based on how much
        # information is preserved after sanitization
        with torch.no_grad():
            original_conf = torch.softmax(output, dim=-1).max().item()
            sanitized_conf = torch.softmax(sanitized_output, dim=-1).max().item()
            
            # Higher preserved confidence = higher attack success
            conf_ratio = sanitized_conf / (original_conf + 1e-8)
            
            # Add noise to simulate attack uncertainty
            attack_success = conf_ratio * (0.8 + 0.4 * np.random.random())
            
        return float(np.clip(attack_success, 0, 1))
    
    def compute_utility(
        self,
        output: torch.Tensor,
        sanitized_output: torch.Tensor
    ) -> float:
        """
        Compute utility preservation score.
        
        Returns how much useful information is preserved.
        """
        with torch.no_grad():
            # Check if predictions match
            orig_pred = output.argmax(dim=-1)
            san_pred = sanitized_output.argmax(dim=-1)
            
            accuracy_preserved = (orig_pred == san_pred).float().mean().item()
            
            # Also check confidence preservation
            orig_conf = torch.softmax(output, dim=-1).max(dim=-1).values
            san_conf = torch.softmax(sanitized_output, dim=-1).max(dim=-1).values
            
            conf_ratio = (san_conf / (orig_conf + 1e-8)).mean().item()
            
            utility = 0.7 * accuracy_preserved + 0.3 * conf_ratio
            
        return float(np.clip(utility, 0, 1))


class RAPTORFramework(nn.Module):
    """
    RAPTOR: Real-time Adaptive Privacy Through Output Regulation
    
    Main framework class providing:
    1. Real-time privacy risk quantification (RPLQ)
    2. Adaptive output sanitization (AOS) 
    3. Online Pareto optimization (PUPO)
    4. Closed-loop feedback for continuous improvement
    
    Usage:
        raptor = RAPTORFramework(model)
        
        # Wrap inference
        @raptor.protect
        def inference(x):
            return model(x)
        
        # Or manual usage
        output = model(query)
        protected_output = raptor.process(query, output)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[Union[RAPTORConfig, str]] = None,
        shadow_model: Optional[nn.Module] = None,
        training_embeddings: Optional[np.ndarray] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        # Load config
        if isinstance(config, str):
            self.config = RAPTORConfig.from_yaml(config)
        elif config is None:
            self.config = RAPTORConfig()
        else:
            self.config = config
        
        self.device = device
        self.model = model.to(device)
        
        # Initialize components
        self.rplq = RPLQ(
            model=model,
            config=self.config.rplq,
            shadow_model=shadow_model,
            training_embeddings=training_embeddings,
            device=device
        )
        
        self.aos = AOS(config=self.config.aos)
        
        self.pupo = PUPO(config=self.config.pupo)
        
        # Attack simulator for feedback (would be real monitoring in production)
        self.attack_simulator = AttackSimulator(model) if self.config.simulate_attacks else None
        
        # Query logging
        self.query_logger = QueryLogger() if self.config.log_queries else None
        self.query_count = 0
        
        logger.info("RAPTOR Framework initialized")
        logger.info(f"  - RPLQ: α={self.config.rplq.alpha}, β={self.config.rplq.beta}, γ={self.config.rplq.gamma}")
        logger.info(f"  - AOS: σ∈[{self.config.aos.sigma_min}, {self.config.aos.sigma_max}]")
        logger.info(f"  - PUPO: λ_privacy={self.config.pupo.lambda_privacy}")
    
    def process(
        self,
        query: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Process a query through the RAPTOR pipeline.
        
        Args:
            query: Input query
            output: Pre-computed model output (computed if not provided)
            return_metrics: Whether to return risk metrics
        
        Returns:
            Sanitized output (and metrics if requested)
        """
        start_time = time.time()
        
        self.query_count += 1
        query = query.to(self.device)
        
        # Get model output if not provided
        if output is None:
            self.model.eval()
            with torch.no_grad():
                output = self.model(query)
                if isinstance(output, tuple):
                    output = output[0]
        
        output = output.to(self.device)
        
        # Step 1: RPLQ - Compute privacy risk
        risk_scores = self.rplq.compute_risk(query, output)
        rho = risk_scores['rho']
        
        # Step 2: Get optimal noise level from PUPO
        if self.config.enable_feedback_loop:
            sigma = self.pupo.get_optimal_sigma(rho)
        else:
            # Use default mapping
            sigma = self.config.aos.sigma_min + rho * (
                self.config.aos.sigma_max - self.config.aos.sigma_min
            )
        
        # Step 3: AOS - Apply adaptive sanitization
        sanitized_output = self.aos.sanitize(output, rho)
        
        # Step 4: Feedback loop (if enabled)
        if self.config.enable_feedback_loop and self.attack_simulator is not None:
            # Simulate attack to get feedback
            if np.random.random() < self.config.attack_sample_rate:
                attack_success = self.attack_simulator.simulate_mia(
                    query, output, sanitized_output
                )
                utility = self.attack_simulator.compute_utility(
                    output, sanitized_output
                )
                
                # Update PUPO with feedback
                self.pupo.record_experience(rho, attack_success, utility, sigma)
        
        # Log query
        latency_ms = (time.time() - start_time) * 1000
        
        if self.query_logger is not None:
            self.query_logger.log(
                self.query_count,
                risk_scores,
                sigma,
                latency_ms
            )
        
        if return_metrics:
            metrics = {
                **risk_scores,
                'sigma': sigma,
                'latency_ms': latency_ms,
                'epsilon': self.aos.get_epsilon(rho)
            }
            return sanitized_output, metrics
        
        return sanitized_output
    
    def protect(self, func: Callable) -> Callable:
        """
        Decorator to protect model inference.
        
        Usage:
            @raptor.protect
            def my_inference(x):
                return model(x)
        """
        @wraps(func)
        def wrapper(query, *args, **kwargs):
            # Get original output
            output = func(query, *args, **kwargs)
            
            # Process through RAPTOR
            return self.process(query, output)
        
        return wrapper
    
    def calibrate(
        self,
        member_samples: torch.Tensor,
        non_member_samples: torch.Tensor
    ):
        """
        Calibrate RPLQ with known member/non-member samples.
        
        This improves MIA risk estimation accuracy.
        """
        self.rplq.calibrate(member_samples, non_member_samples)
        logger.info("RAPTOR calibrated with member/non-member samples")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get framework statistics."""
        stats = {
            'total_queries': self.query_count,
            'pupo': self.pupo.get_statistics()
        }
        
        if self.query_logger is not None:
            stats['query_stats'] = self.query_logger.get_statistics()
        
        return stats
    
    def get_privacy_guarantee(self) -> Dict[str, float]:
        """
        Get current privacy guarantee under composition.
        
        Based on Theorem 1 in the paper.
        """
        if self.query_logger is None or len(self.query_logger.entries) == 0:
            return {'epsilon_total': 0.0, 'num_queries': 0}
        
        risks = [e['rho'] for e in self.query_logger.entries]
        k = len(risks)
        
        # Get max epsilon from all queries
        max_epsilon = max(self.aos.get_epsilon(r) for r in risks)
        
        # Advanced composition (Theorem 1)
        delta = 1e-5  # Standard choice
        epsilon_total = (
            np.sqrt(2 * k * np.log(1/delta)) * max_epsilon +
            k * (np.exp(max_epsilon) - 1) / (np.exp(max_epsilon) + 1) * max_epsilon
        )
        
        return {
            'epsilon_total': epsilon_total,
            'max_per_query_epsilon': max_epsilon,
            'num_queries': k,
            'delta': delta
        }
    
    def forward(
        self,
        query: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for nn.Module compatibility."""
        return self.process(query)
    
    def save(self, path: str):
        """Save framework state."""
        state = {
            'config': self.config,
            'rplq_state': self.rplq.state_dict(),
            'aos_state': self.aos.state_dict(),
            'pupo_state': self.pupo.state_dict(),
            'query_count': self.query_count
        }
        torch.save(state, path)
        logger.info(f"RAPTOR state saved to {path}")
    
    def load(self, path: str):
        """Load framework state."""
        state = torch.load(path)
        self.config = state['config']
        self.rplq.load_state_dict(state['rplq_state'])
        self.aos.load_state_dict(state['aos_state'])
        self.pupo.load_state_dict(state['pupo_state'])
        self.query_count = state['query_count']
        logger.info(f"RAPTOR state loaded from {path}")


# Convenience function
def create_raptor(
    model: nn.Module,
    sigma_range: tuple = (0.01, 1.0),
    lambda_privacy: float = 0.5,
    **kwargs
) -> RAPTORFramework:
    """
    Factory function to create RAPTOR with common settings.
    
    Args:
        model: ML model to protect
        sigma_range: (min, max) noise range
        lambda_privacy: Weight for privacy objective
        **kwargs: Additional config options
    
    Returns:
        Configured RAPTOR instance
    """
    config = RAPTORConfig()
    config.aos.sigma_min = sigma_range[0]
    config.aos.sigma_max = sigma_range[1]
    config.pupo.lambda_privacy = lambda_privacy
    config.pupo.lambda_utility = 1 - lambda_privacy
    
    return RAPTORFramework(model=model, config=config, **kwargs)
