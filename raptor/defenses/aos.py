"""
AOS: Adaptive Output Sanitization

This module implements multi-layer output sanitization that dynamically
calibrates defense strength based on measured privacy risk.

References:
    - Kulynych et al. (NeurIPS 2024) "Attack-Aware Noise Calibration for DP"
    - Jia et al. (CCS 2019) "MemGuard"
    - Dwork et al. "Differential Privacy"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AOSConfig:
    """Configuration for AOS module."""
    # Noise parameters
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    noise_type: str = "gaussian"  # gaussian, laplace
    
    # Temperature scaling
    temperature_min: float = 1.0
    temperature_max: float = 3.0
    
    # Truncation
    truncation_threshold: float = 0.7
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Risk mapping
    mapping_type: str = "sigmoid"  # linear, sigmoid, learned
    sigmoid_steepness: float = 10.0
    sigmoid_midpoint: float = 0.5
    
    # Output domain
    clip_outputs: bool = True
    output_min: float = 0.0
    output_max: float = 1.0


class RiskToNoiseMapping(nn.Module):
    """
    Maps privacy risk ρ to noise scale σ.
    
    σ(ρ) = σ_min + (σ_max - σ_min) · g(ρ)
    
    where g: [0,1] → [0,1] is a monotonic mapping function.
    """
    
    def __init__(self, config: AOSConfig):
        super().__init__()
        self.config = config
        
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        
        if config.mapping_type == "learned":
            # Learnable monotonic mapping using cumulative softmax
            self.mapping_params = nn.Parameter(torch.zeros(10))
        
    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """Map risk to noise scale."""
        if self.config.mapping_type == "linear":
            g = rho
        elif self.config.mapping_type == "sigmoid":
            # Sigmoid mapping for sharper transition
            k = self.config.sigmoid_steepness
            x0 = self.config.sigmoid_midpoint
            g = torch.sigmoid(k * (rho - x0))
        elif self.config.mapping_type == "learned":
            # Monotonic learned mapping using cumulative softmax
            weights = F.softmax(self.mapping_params, dim=0)
            cumsum = torch.cumsum(weights, dim=0)
            # Interpolate based on rho
            idx = (rho * (len(cumsum) - 1)).long().clamp(0, len(cumsum) - 1)
            g = cumsum[idx]
        else:
            g = rho
        
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * g
        return sigma


class ConfidencePerturbation(nn.Module):
    """
    Perturbs confidence scores using temperature scaling.
    
    Higher risk → higher temperature → flatter distribution → less information
    """
    
    def __init__(self, temp_min: float = 1.0, temp_max: float = 3.0):
        super().__init__()
        self.temp_min = temp_min
        self.temp_max = temp_max
    
    def forward(
        self,
        logits: torch.Tensor,
        risk: float
    ) -> torch.Tensor:
        """
        Apply temperature scaling based on risk.
        
        Args:
            logits: Model logits (before softmax)
            risk: Privacy risk in [0, 1]
        
        Returns:
            Temperature-scaled logits
        """
        # Higher risk → higher temperature
        temperature = self.temp_min + risk * (self.temp_max - self.temp_min)
        
        scaled_logits = logits / temperature
        
        return scaled_logits


class CalibratedNoiseInjection(nn.Module):
    """
    Injects calibrated noise based on privacy risk.
    
    Based on attack-aware noise calibration (Kulynych et al., 2024)
    """
    
    def __init__(
        self,
        noise_type: str = "gaussian",
        sigma_min: float = 0.01,
        sigma_max: float = 1.0
    ):
        super().__init__()
        self.noise_type = noise_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def forward(
        self,
        output: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        Add calibrated noise to output.
        
        Args:
            output: Model output tensor
            sigma: Noise scale (from risk mapping)
        
        Returns:
            Noisy output
        """
        if sigma < 1e-8:
            return output
        
        if self.noise_type == "gaussian":
            noise = torch.randn_like(output) * sigma
        elif self.noise_type == "laplace":
            # Laplace noise for DP
            uniform = torch.rand_like(output) - 0.5
            noise = -sigma * torch.sign(uniform) * torch.log(1 - 2 * torch.abs(uniform))
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return output + noise


class SelectiveTruncation(nn.Module):
    """
    Selectively truncates/masks high-risk tokens or outputs.
    
    For classification: top-k/top-p filtering
    For generation: token-level risk masking
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        super().__init__()
        self.threshold = threshold
        self.top_k = top_k
        self.top_p = top_p
    
    def forward(
        self,
        output: torch.Tensor,
        risk: float,
        token_risks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply selective truncation based on risk.
        
        Args:
            output: Model output (logits or probabilities)
            risk: Overall privacy risk
            token_risks: Per-token risk scores (for LLMs)
        
        Returns:
            Truncated output
        """
        if risk < self.threshold:
            # Low risk: minimal truncation
            return output
        
        # Apply top-k filtering if specified
        if self.top_k is not None and output.dim() >= 2:
            # Keep only top-k values
            topk_values, _ = torch.topk(output, min(self.top_k, output.size(-1)), dim=-1)
            threshold_value = topk_values[..., -1:]
            output = torch.where(
                output >= threshold_value,
                output,
                torch.full_like(output, float('-inf'))
            )
        
        # Apply top-p (nucleus) filtering
        if self.top_p is not None and output.dim() >= 2:
            sorted_probs, sorted_indices = torch.sort(
                F.softmax(output, dim=-1), descending=True, dim=-1
            )
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff
            mask = cumsum - sorted_probs > self.top_p
            sorted_probs[mask] = 0.0
            
            # Unsort
            output = torch.zeros_like(output).scatter_(-1, sorted_indices, sorted_probs)
            output = torch.log(output + 1e-10)
        
        # Token-level masking for LLMs
        if token_risks is not None:
            # Mask tokens with high individual risk
            high_risk_mask = token_risks > self.threshold
            if output.dim() >= 2:
                output = output.masked_fill(
                    high_risk_mask.unsqueeze(-1),
                    float('-inf')
                )
        
        return output


class AOS(nn.Module):
    """
    Adaptive Output Sanitization (AOS)
    
    Multi-layer defense: S_ρ(y) = T_ρ ∘ N_ρ ∘ P_ρ(y)
    
    This is Algorithm 2 in the RAPTOR paper.
    """
    
    def __init__(
        self,
        config: Optional[AOSConfig] = None
    ):
        super().__init__()
        
        self.config = config or AOSConfig()
        
        # Initialize components
        self.risk_mapping = RiskToNoiseMapping(self.config)
        
        self.confidence_perturbation = ConfidencePerturbation(
            temp_min=self.config.temperature_min,
            temp_max=self.config.temperature_max
        )
        
        self.noise_injection = CalibratedNoiseInjection(
            noise_type=self.config.noise_type,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max
        )
        
        self.truncation = SelectiveTruncation(
            threshold=self.config.truncation_threshold,
            top_k=self.config.top_k,
            top_p=self.config.top_p
        )
        
        logger.info(f"AOS initialized: σ∈[{self.config.sigma_min}, {self.config.sigma_max}], "
                   f"T∈[{self.config.temperature_min}, {self.config.temperature_max}]")
    
    def sanitize(
        self,
        output: torch.Tensor,
        risk: float,
        is_logits: bool = True,
        token_risks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-layer sanitization to model output.
        
        Args:
            output: Model output tensor
            risk: Privacy risk score in [0, 1]
            is_logits: Whether output is logits (pre-softmax)
            token_risks: Per-token risk scores (for LLMs)
        
        Returns:
            Sanitized output tensor
        """
        sanitized = output.clone()
        
        # Layer 1: Confidence Perturbation (temperature scaling)
        if is_logits:
            sanitized = self.confidence_perturbation(sanitized, risk)
        
        # Layer 2: Calibrated Noise Injection
        sigma = self.risk_mapping(torch.tensor(risk)).item()
        sanitized = self.noise_injection(sanitized, sigma)
        
        # Layer 3: Selective Truncation
        sanitized = self.truncation(sanitized, risk, token_risks)
        
        # Project to valid domain
        if self.config.clip_outputs:
            if is_logits:
                # For logits, apply softmax then clip
                probs = F.softmax(sanitized, dim=-1)
                probs = torch.clamp(probs, self.config.output_min, self.config.output_max)
                # Renormalize
                probs = probs / probs.sum(dim=-1, keepdim=True)
                sanitized = torch.log(probs + 1e-10)
            else:
                sanitized = torch.clamp(
                    sanitized,
                    self.config.output_min,
                    self.config.output_max
                )
        
        return sanitized
    
    def get_epsilon(self, risk: float) -> float:
        """
        Compute DP epsilon guarantee for given risk level.
        
        ε(ρ) = Δf / σ(ρ)
        
        Assumes sensitivity Δf = 1 (can be scaled for specific applications)
        """
        sigma = self.risk_mapping(torch.tensor(risk)).item()
        
        if sigma < 1e-8:
            return float('inf')
        
        # For Gaussian mechanism
        epsilon = 1.0 / sigma  # Simplified; full computation requires δ
        
        return epsilon
    
    def forward(
        self,
        output: torch.Tensor,
        risk: float
    ) -> torch.Tensor:
        """Forward pass for nn.Module compatibility."""
        return self.sanitize(output, risk)
    
    def update_mapping(self, new_params: torch.Tensor):
        """Update risk-to-noise mapping parameters (called by PUPO)."""
        if self.config.mapping_type == "learned":
            self.risk_mapping.mapping_params.data = new_params


class TokenLevelAOS(AOS):
    """
    Token-level AOS for Large Language Models.
    
    Extends AOS with per-token risk assessment and sanitization.
    """
    
    def __init__(
        self,
        config: Optional[AOSConfig] = None,
        tokenizer: Optional[Any] = None
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        
        # PII patterns for token-level risk
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
    
    def compute_token_risks(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token privacy risk."""
        batch_size, seq_len = tokens.shape[:2]
        token_risks = torch.zeros(batch_size, seq_len, device=tokens.device)
        
        # Risk based on token confidence
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        
        # High confidence → potential memorization
        token_risks = torch.where(
            max_probs > 0.9,
            torch.ones_like(token_risks) * 0.8,
            token_risks
        )
        
        return token_risks
    
    def sanitize_tokens(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        overall_risk: float
    ) -> torch.Tensor:
        """
        Apply token-level sanitization for LLMs.
        
        Args:
            logits: Next-token logits [batch, seq, vocab]
            tokens: Current token IDs [batch, seq]
            overall_risk: Overall sequence risk
        
        Returns:
            Sanitized logits
        """
        # Compute per-token risks
        token_risks = self.compute_token_risks(tokens, logits)
        
        # Apply base sanitization with token-level info
        sanitized = self.sanitize(
            logits,
            overall_risk,
            is_logits=True,
            token_risks=token_risks
        )
        
        return sanitized


# Utility functions
def create_aos(
    sigma_range: Tuple[float, float] = (0.01, 1.0),
    temperature_range: Tuple[float, float] = (1.0, 3.0),
    truncation_threshold: float = 0.7,
    noise_type: str = "gaussian"
) -> AOS:
    """
    Factory function to create AOS with common configurations.
    
    Args:
        sigma_range: (min, max) noise scale
        temperature_range: (min, max) temperature
        truncation_threshold: Risk threshold for truncation
        noise_type: "gaussian" or "laplace"
    
    Returns:
        Configured AOS instance
    """
    config = AOSConfig(
        sigma_min=sigma_range[0],
        sigma_max=sigma_range[1],
        temperature_min=temperature_range[0],
        temperature_max=temperature_range[1],
        truncation_threshold=truncation_threshold,
        noise_type=noise_type
    )
    
    return AOS(config)
