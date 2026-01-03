"""
RPLQ: Real-time Privacy Leakage Quantification

This module implements the core privacy risk scoring algorithm that combines
membership inference, model inversion, and extraction risks into a unified score.

References:
    - Carlini et al. (2022) "Membership Inference Attacks From First Principles" (LiRA)
    - Murakonda & Shokri (2020) "ML Privacy Meter"
    - Wen et al. (2025) "Understanding Data Importance in ML Attacks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RPLQConfig:
    """Configuration for RPLQ module."""
    alpha: float = 0.4  # Weight for membership inference risk
    beta: float = 0.3   # Weight for model inversion risk
    gamma: float = 0.3  # Weight for extraction risk
    
    # MIA parameters
    num_shadow_models: int = 1  # Use 1 for efficiency, increase for accuracy
    mia_threshold: float = 0.5
    
    # Inversion parameters
    similarity_k: int = 10  # k-NN for similarity
    entropy_epsilon: float = 1e-8
    
    # Extraction parameters
    perplexity_threshold: float = 50.0
    repetition_window: int = 5
    
    # Importance network
    use_importance_net: bool = True
    importance_hidden_dim: int = 128


class ImportanceNetwork(nn.Module):
    """
    Neural network to estimate sample-specific vulnerability (ψ in the paper).
    Inspired by CaliMem from Wen et al. (2025).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MembershipInferenceRisk:
    """
    Computes membership inference risk using LiRA-inspired likelihood ratio.
    
    Based on Carlini et al. (2022) "Membership Inference Attacks From First Principles"
    """
    
    def __init__(
        self,
        model: nn.Module,
        shadow_model: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.shadow_model = shadow_model
        self.device = device
        
        # Statistics for IN/OUT distributions (computed during calibration)
        self.in_mean = 0.0
        self.in_std = 1.0
        self.out_mean = 0.0
        self.out_std = 1.0
        self.calibrated = False
    
    def calibrate(
        self,
        member_samples: torch.Tensor,
        non_member_samples: torch.Tensor
    ):
        """Calibrate the attack using known member/non-member samples."""
        self.model.eval()
        
        with torch.no_grad():
            # Get logits for members
            member_logits = self._get_confidence_logits(member_samples)
            non_member_logits = self._get_confidence_logits(non_member_samples)
        
        # Fit Gaussian distributions (LiRA approach)
        self.in_mean = member_logits.mean().item()
        self.in_std = member_logits.std().item() + 1e-8
        self.out_mean = non_member_logits.mean().item()
        self.out_std = non_member_logits.std().item() + 1e-8
        self.calibrated = True
        
        logger.info(f"MIA calibrated: IN(μ={self.in_mean:.3f}, σ={self.in_std:.3f}), "
                   f"OUT(μ={self.out_mean:.3f}, σ={self.out_std:.3f})")
    
    def _get_confidence_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get confidence scores in logit space."""
        x = x.to(self.device)
        outputs = self.model(x)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Get max confidence
        probs = F.softmax(outputs, dim=-1)
        max_conf = probs.max(dim=-1).values
        
        # Convert to logit space for Gaussian fitting
        logits = torch.log(max_conf / (1 - max_conf + 1e-8))
        return logits
    
    def compute_risk(
        self,
        query: torch.Tensor,
        output: torch.Tensor
    ) -> float:
        """
        Compute membership inference risk for a query-output pair.
        
        Returns:
            float: Risk score in [0, 1], higher means more likely to be member
        """
        # Get confidence in logit space
        if output.dim() > 1:
            probs = F.softmax(output, dim=-1)
            max_conf = probs.max(dim=-1).values.mean()
        else:
            max_conf = output.max()
        
        conf_logit = torch.log(max_conf / (1 - max_conf + 1e-8)).item()
        
        if not self.calibrated:
            # Without calibration, use heuristic based on confidence
            return float(torch.sigmoid(torch.tensor(conf_logit - 0.5)).item())
        
        # LiRA-style likelihood ratio test
        from scipy.stats import norm
        
        p_in = norm.pdf(conf_logit, self.in_mean, self.in_std)
        p_out = norm.pdf(conf_logit, self.out_mean, self.out_std)
        
        # Likelihood ratio
        lr = p_in / (p_out + 1e-10)
        
        # Convert to probability via sigmoid
        risk = 1 / (1 + np.exp(-np.log(lr + 1e-10)))
        
        return float(np.clip(risk, 0, 1))


class ModelInversionRisk:
    """
    Computes model inversion risk based on query-training similarity and output entropy.
    
    Based on Yang et al. (2025) survey and FSInfo from Deng et al. (2025)
    """
    
    def __init__(
        self,
        training_embeddings: Optional[np.ndarray] = None,
        k: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.k = k
        self.device = device
        self.index = None
        
        if training_embeddings is not None:
            self._build_index(training_embeddings)
    
    def _build_index(self, embeddings: np.ndarray):
        """Build FAISS index for efficient similarity search."""
        try:
            import faiss
            
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)  # Inner product (cosine with normalized)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype(np.float32))
            
            logger.info(f"Built FAISS index with {embeddings.shape[0]} vectors")
        except ImportError:
            logger.warning("FAISS not available, using numpy for similarity")
            self.embeddings = embeddings
    
    def compute_risk(
        self,
        query_embedding: torch.Tensor,
        output: torch.Tensor
    ) -> float:
        """
        Compute model inversion risk.
        
        φ_Inv(q) = max_{x ∈ D} sim(q, x) / (H(y) + ε)
        
        High similarity to training + low entropy = high inversion risk
        """
        # Compute max similarity to training data
        query_np = query_embedding.cpu().numpy().reshape(1, -1).astype(np.float32)
        
        if self.index is not None:
            import faiss
            faiss.normalize_L2(query_np)
            D, _ = self.index.search(query_np, self.k)
            max_sim = D[0, 0]  # Highest similarity
        elif hasattr(self, 'embeddings'):
            # Fallback to numpy
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(query_np, self.embeddings)[0]
            max_sim = np.max(sims)
        else:
            # No training data available
            max_sim = 0.5
        
        # Compute output entropy
        if output.dim() > 1:
            probs = F.softmax(output, dim=-1)
        else:
            probs = output
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
        
        # Normalize entropy (assuming max entropy of log(num_classes))
        max_entropy = np.log(probs.shape[-1]) if probs.dim() > 0 else 1.0
        normalized_entropy = entropy / (max_entropy + 1e-8)
        
        # Risk: high similarity, low entropy
        risk = max_sim / (normalized_entropy + 0.1)
        
        # Normalize to [0, 1]
        risk = float(np.clip(risk / 10, 0, 1))  # Heuristic scaling
        
        return risk


class ExtractionRisk:
    """
    Computes data extraction risk based on output perplexity and repetition.
    
    Based on Nasr et al. (2025) "Scalable Extraction of Training Data"
    """
    
    def __init__(
        self,
        perplexity_threshold: float = 50.0,
        repetition_window: int = 5
    ):
        self.perplexity_threshold = perplexity_threshold
        self.repetition_window = repetition_window
    
    def _compute_repetition_score(self, tokens: torch.Tensor) -> float:
        """Compute n-gram repetition score."""
        if tokens.dim() == 0 or tokens.numel() < self.repetition_window:
            return 0.0
        
        tokens_list = tokens.cpu().tolist()
        if isinstance(tokens_list[0], list):
            tokens_list = tokens_list[0]
        
        # Count n-gram repetitions
        ngrams = []
        for i in range(len(tokens_list) - self.repetition_window + 1):
            ngram = tuple(tokens_list[i:i + self.repetition_window])
            ngrams.append(ngram)
        
        if len(ngrams) == 0:
            return 0.0
        
        unique_ratio = len(set(ngrams)) / len(ngrams)
        repetition_score = 1 - unique_ratio
        
        return repetition_score
    
    def compute_risk(
        self,
        output_logits: torch.Tensor,
        output_tokens: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute extraction risk.
        
        φ_Ext(q) = I[perplexity < τ] · RepScore(y)
        
        Low perplexity + high repetition = memorized content
        """
        # Compute perplexity
        if output_logits.dim() >= 2:
            probs = F.softmax(output_logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            
            # Approximate perplexity from confidence
            log_probs = torch.log(max_probs + 1e-8)
            perplexity = torch.exp(-log_probs.mean()).item()
        else:
            perplexity = self.perplexity_threshold * 2  # Default high
        
        # Check perplexity threshold
        low_perplexity = float(perplexity < self.perplexity_threshold)
        
        # Compute repetition score
        if output_tokens is not None:
            rep_score = self._compute_repetition_score(output_tokens)
        else:
            rep_score = 0.0
        
        # Combined risk
        risk = low_perplexity * (0.5 + 0.5 * rep_score)
        
        return float(np.clip(risk, 0, 1))


class RPLQ(nn.Module):
    """
    Real-time Privacy Leakage Quantification (RPLQ)
    
    Computes unified privacy risk score:
    ρ(q) = α·φ_MI(q) + β·φ_Inv(q) + γ·φ_Ext(q)
    
    This is Algorithm 1 in the RAPTOR paper.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[RPLQConfig] = None,
        shadow_model: Optional[nn.Module] = None,
        training_embeddings: Optional[np.ndarray] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.config = config or RPLQConfig()
        self.device = device
        self.model = model.to(device)
        
        # Initialize risk components
        self.mia_risk = MembershipInferenceRisk(
            model=model,
            shadow_model=shadow_model,
            device=device
        )
        
        self.inversion_risk = ModelInversionRisk(
            training_embeddings=training_embeddings,
            k=self.config.similarity_k,
            device=device
        )
        
        self.extraction_risk = ExtractionRisk(
            perplexity_threshold=self.config.perplexity_threshold,
            repetition_window=self.config.repetition_window
        )
        
        # Importance network (optional)
        if self.config.use_importance_net:
            # Will be initialized when input dim is known
            self.importance_net = None
        
        # Learnable weights (for PUPO updates)
        self.register_buffer('alpha', torch.tensor(self.config.alpha))
        self.register_buffer('beta', torch.tensor(self.config.beta))
        self.register_buffer('gamma', torch.tensor(self.config.gamma))
        
        logger.info(f"RPLQ initialized with weights: α={self.config.alpha}, "
                   f"β={self.config.beta}, γ={self.config.gamma}")
    
    def calibrate(
        self,
        member_samples: torch.Tensor,
        non_member_samples: torch.Tensor
    ):
        """Calibrate MIA component with known member/non-member samples."""
        self.mia_risk.calibrate(member_samples, non_member_samples)
    
    def compute_risk(
        self,
        query: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        query_embedding: Optional[torch.Tensor] = None,
        output_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute unified privacy risk score for a query.
        
        Args:
            query: Input query tensor
            output: Model output (computed if not provided)
            query_embedding: Embedding for similarity search
            output_tokens: Token IDs for extraction risk (LLMs)
        
        Returns:
            Dict with individual risks and unified score ρ
        """
        self.model.eval()
        
        with torch.no_grad():
            query = query.to(self.device)
            
            # Get model output if not provided
            if output is None:
                output = self.model(query)
                if isinstance(output, tuple):
                    output = output[0]
            
            output = output.to(self.device)
            
            # Compute individual risk components
            phi_mi = self.mia_risk.compute_risk(query, output)
            
            # Use query as embedding if not provided
            if query_embedding is None:
                query_embedding = query.flatten()
            
            phi_inv = self.inversion_risk.compute_risk(query_embedding, output)
            phi_ext = self.extraction_risk.compute_risk(output, output_tokens)
            
            # Compute importance weight (ψ)
            psi = 1.0
            if self.importance_net is not None:
                psi = self.importance_net(query.flatten().unsqueeze(0)).item()
            
            # Apply importance to MIA (as in Wen et al.)
            phi_mi = phi_mi * psi
            
            # Unified risk score
            rho = (
                self.alpha.item() * phi_mi +
                self.beta.item() * phi_inv +
                self.gamma.item() * phi_ext
            )
            
            # Clamp to [0, 1]
            rho = float(np.clip(rho, 0, 1))
        
        return {
            'rho': rho,
            'phi_mi': phi_mi,
            'phi_inv': phi_inv,
            'phi_ext': phi_ext,
            'psi': psi
        }
    
    def update_weights(self, alpha: float, beta: float, gamma: float):
        """Update risk aggregation weights (called by PUPO)."""
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1"
        
        self.alpha.fill_(alpha)
        self.beta.fill_(beta)
        self.gamma.fill_(gamma)
    
    def forward(
        self,
        query: torch.Tensor,
        output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass returning risk score as tensor."""
        result = self.compute_risk(query, output)
        return torch.tensor(result['rho'], device=self.device)


# Convenience function for quick risk assessment
def compute_privacy_risk(
    model: nn.Module,
    query: torch.Tensor,
    config: Optional[RPLQConfig] = None
) -> float:
    """
    Convenience function to compute privacy risk for a single query.
    
    Args:
        model: The ML model
        query: Input query
        config: Optional RPLQ configuration
    
    Returns:
        Privacy risk score in [0, 1]
    """
    rplq = RPLQ(model=model, config=config)
    result = rplq.compute_risk(query)
    return result['rho']
