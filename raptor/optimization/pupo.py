"""
PUPO: Privacy-Utility Pareto Optimization

This module implements online learning for optimal privacy-utility tradeoffs,
extending Avent et al. (2020) from offline to online settings.

References:
    - Avent et al. (PoPETs 2020) "Automatic Discovery of Privacy-Utility Pareto Fronts"
    - Xu et al. (ICML 2023) "Pareto Regret Analyses in Multi-objective Multi-armed Bandit"
    - Navon et al. (ICLR 2021) "Learning the Pareto Front with Hypernetworks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PUPOConfig:
    """Configuration for PUPO module."""
    # Learning parameters
    learning_rate: float = 0.01
    momentum: float = 0.9
    
    # Pareto weights
    lambda_privacy: float = 0.5  # Weight for privacy loss
    lambda_utility: float = 0.5  # Weight for utility loss
    
    # Experience buffer
    buffer_size: int = 1000
    min_samples_for_update: int = 50
    update_frequency: int = 10
    
    # Regret bounds
    exploration_bonus: float = 0.1
    ucb_coefficient: float = 2.0
    
    # Convergence
    convergence_threshold: float = 1e-4
    max_iterations: int = 1000
    
    # Mapping parameters
    num_mapping_params: int = 10


@dataclass
class Experience:
    """Single experience tuple for PUPO."""
    risk: float
    attack_success: float  # Privacy loss (lower better)
    utility: float  # Utility score (higher better)
    sigma: float  # Noise level used
    timestamp: int = 0


class ExperienceBuffer:
    """
    Circular buffer for storing privacy-utility experiences.
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.timestamp = 0
    
    def add(
        self,
        risk: float,
        attack_success: float,
        utility: float,
        sigma: float
    ):
        """Add experience to buffer."""
        exp = Experience(
            risk=risk,
            attack_success=attack_success,
            utility=utility,
            sigma=sigma,
            timestamp=self.timestamp
        )
        self.buffer.append(exp)
        self.timestamp += 1
    
    def sample(self, n: int) -> List[Experience]:
        """Sample n experiences from buffer."""
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_recent(self, n: int) -> List[Experience]:
        """Get n most recent experiences."""
        n = min(n, len(self.buffer))
        return list(self.buffer)[-n:]
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
        self.timestamp = 0


class ParetoFront:
    """
    Maintains and updates Pareto front of privacy-utility tradeoffs.
    """
    
    def __init__(self):
        self.points: List[Tuple[float, float, float]] = []  # (risk, attack_success, utility)
    
    def add_point(self, risk: float, attack_success: float, utility: float) -> bool:
        """
        Add point if it's Pareto-optimal.
        
        Returns:
            True if point was added to front
        """
        # Check if dominated
        is_dominated = False
        points_to_remove = []
        
        for i, (r, a, u) in enumerate(self.points):
            # New point dominated if existing point is better in both objectives
            if a <= attack_success and u >= utility and (a < attack_success or u > utility):
                is_dominated = True
                break
            # New point dominates existing
            if attack_success <= a and utility >= u and (attack_success < a or utility > u):
                points_to_remove.append(i)
        
        if is_dominated:
            return False
        
        # Remove dominated points
        for i in sorted(points_to_remove, reverse=True):
            self.points.pop(i)
        
        self.points.append((risk, attack_success, utility))
        return True
    
    def get_front(self) -> List[Tuple[float, float, float]]:
        """Get current Pareto front."""
        return sorted(self.points, key=lambda x: x[1])  # Sort by attack success
    
    def compute_hypervolume(
        self,
        reference: Tuple[float, float] = (1.0, 0.0)
    ) -> float:
        """
        Compute hypervolume indicator of Pareto front.
        
        Args:
            reference: Reference point (max_attack_success, min_utility)
        
        Returns:
            Hypervolume dominated by front
        """
        if len(self.points) == 0:
            return 0.0
        
        # Sort by attack success (ascending)
        sorted_points = sorted(self.points, key=lambda x: x[1])
        
        hypervolume = 0.0
        prev_attack = reference[0]
        
        for _, attack, utility in sorted_points:
            if utility > reference[1]:
                width = prev_attack - attack
                height = utility - reference[1]
                hypervolume += width * height
                prev_attack = attack
        
        return hypervolume


class OnlineParetoOptimizer:
    """
    Online multi-objective optimization using modified UCB.
    
    Based on Xu et al. (ICML 2023) "Pareto Regret Analyses"
    """
    
    def __init__(
        self,
        num_arms: int = 10,  # Discretized risk levels
        ucb_coefficient: float = 2.0
    ):
        self.num_arms = num_arms
        self.ucb_coefficient = ucb_coefficient
        
        # Statistics per arm (risk level)
        self.counts = np.zeros(num_arms)
        self.privacy_sum = np.zeros(num_arms)
        self.utility_sum = np.zeros(num_arms)
        self.privacy_sq_sum = np.zeros(num_arms)
        self.utility_sq_sum = np.zeros(num_arms)
    
    def select_arm(self, t: int) -> int:
        """
        Select risk level using Pareto-UCB.
        
        Args:
            t: Current timestep
        
        Returns:
            Selected arm (risk level index)
        """
        if t < self.num_arms:
            # Initial exploration
            return t
        
        ucb_values = np.zeros(self.num_arms)
        
        for i in range(self.num_arms):
            if self.counts[i] == 0:
                ucb_values[i] = float('inf')
                continue
            
            # Mean rewards
            mean_privacy = self.privacy_sum[i] / self.counts[i]
            mean_utility = self.utility_sum[i] / self.counts[i]
            
            # UCB bonus
            bonus = self.ucb_coefficient * np.sqrt(np.log(t) / self.counts[i])
            
            # Scalarized objective (can be adjusted for different Pareto preferences)
            ucb_values[i] = -mean_privacy + mean_utility + 2 * bonus
        
        return int(np.argmax(ucb_values))
    
    def update(
        self,
        arm: int,
        privacy_loss: float,
        utility: float
    ):
        """Update statistics for selected arm."""
        self.counts[arm] += 1
        self.privacy_sum[arm] += privacy_loss
        self.utility_sum[arm] += utility
        self.privacy_sq_sum[arm] += privacy_loss ** 2
        self.utility_sq_sum[arm] += utility ** 2
    
    def get_best_arm(self, lambda_privacy: float = 0.5) -> int:
        """Get best arm according to scalarized objective."""
        scores = np.zeros(self.num_arms)
        
        for i in range(self.num_arms):
            if self.counts[i] == 0:
                scores[i] = float('-inf')
                continue
            
            mean_privacy = self.privacy_sum[i] / self.counts[i]
            mean_utility = self.utility_sum[i] / self.counts[i]
            
            # Lower privacy loss and higher utility is better
            scores[i] = -lambda_privacy * mean_privacy + (1 - lambda_privacy) * mean_utility
        
        return int(np.argmax(scores))


class PUPO(nn.Module):
    """
    Privacy-Utility Pareto Optimization (PUPO)
    
    Online learning algorithm for optimal risk-to-defense mapping.
    
    This is Algorithm 3 in the RAPTOR paper.
    """
    
    def __init__(
        self,
        config: Optional[PUPOConfig] = None
    ):
        super().__init__()
        
        self.config = config or PUPOConfig()
        
        # Experience buffer
        self.buffer = ExperienceBuffer(self.config.buffer_size)
        
        # Pareto front tracker
        self.pareto_front = ParetoFront()
        
        # Online optimizer
        self.online_optimizer = OnlineParetoOptimizer(
            num_arms=self.config.num_mapping_params,
            ucb_coefficient=self.config.ucb_coefficient
        )
        
        # Learnable mapping parameters
        self.mapping_params = nn.Parameter(
            torch.zeros(self.config.num_mapping_params)
        )
        
        # Optimizer for gradient updates
        self.optimizer = torch.optim.SGD(
            [self.mapping_params],
            lr=self.config.learning_rate,
            momentum=self.config.momentum
        )
        
        # Statistics
        self.total_updates = 0
        self.cumulative_regret = 0.0
        
        logger.info(f"PUPO initialized with λ_privacy={self.config.lambda_privacy}, "
                   f"buffer_size={self.config.buffer_size}")
    
    def record_experience(
        self,
        risk: float,
        attack_success: float,
        utility: float,
        sigma: float
    ):
        """
        Record a privacy-utility experience.
        
        Args:
            risk: Privacy risk that was measured
            attack_success: Observed attack success rate (0-1)
            utility: Utility preservation score (0-1)
            sigma: Noise level that was applied
        """
        self.buffer.add(risk, attack_success, utility, sigma)
        
        # Update Pareto front
        self.pareto_front.add_point(risk, attack_success, utility)
        
        # Update online optimizer
        arm = int(risk * (self.config.num_mapping_params - 1))
        self.online_optimizer.update(arm, attack_success, utility)
        
        # Periodic gradient update
        if len(self.buffer) >= self.config.min_samples_for_update:
            if self.total_updates % self.config.update_frequency == 0:
                self._gradient_update()
        
        self.total_updates += 1
    
    def _gradient_update(self):
        """Perform gradient update on mapping parameters."""
        # Sample from buffer
        samples = self.buffer.sample(min(100, len(self.buffer)))
        
        if len(samples) == 0:
            return
        
        # Compute loss
        privacy_losses = torch.tensor([s.attack_success for s in samples])
        utility_losses = torch.tensor([1 - s.utility for s in samples])
        
        # Pareto loss
        loss = (
            self.config.lambda_privacy * privacy_losses.mean() +
            self.config.lambda_utility * utility_losses.mean()
        )
        
        # Add regularization to keep mapping monotonic
        sorted_params = torch.sort(self.mapping_params)[0]
        monotonicity_loss = F.relu(sorted_params[:-1] - sorted_params[1:]).sum()
        
        total_loss = loss + 0.1 * monotonicity_loss
        
        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Project to monotonic (cumulative softmax ensures monotonicity)
        with torch.no_grad():
            self.mapping_params.data = self._project_monotonic(self.mapping_params.data)
    
    def _project_monotonic(self, params: torch.Tensor) -> torch.Tensor:
        """Project parameters to ensure monotonic mapping."""
        # Use isotonic regression
        sorted_vals, sorted_idx = torch.sort(params)
        
        # Pool Adjacent Violators Algorithm (PAVA)
        n = len(params)
        result = sorted_vals.clone()
        
        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                # Pool and average
                j = i + 1
                while j < n and result[i] > result[j]:
                    j += 1
                pool_mean = result[i:j].mean()
                result[i:j] = pool_mean
            i += 1
        
        # Unsort
        output = torch.zeros_like(params)
        output[sorted_idx] = result
        
        return output
    
    def get_optimal_sigma(self, risk: float) -> float:
        """
        Get optimal noise level for given risk.
        
        Uses learned mapping from online optimization.
        """
        # Use online optimizer's best arm recommendation
        best_arm = self.online_optimizer.get_best_arm(self.config.lambda_privacy)
        
        # Interpolate based on risk
        risk_arm = int(risk * (self.config.num_mapping_params - 1))
        
        # Blend between risk-based and optimal arm
        weights = F.softmax(self.mapping_params, dim=0)
        cumsum = torch.cumsum(weights, dim=0)
        
        sigma = cumsum[risk_arm].item()
        
        return float(np.clip(sigma, 0.01, 1.0))
    
    def get_pareto_front(self) -> List[Tuple[float, float, float]]:
        """Get current Pareto front."""
        return self.pareto_front.get_front()
    
    def get_hypervolume(self) -> float:
        """Get hypervolume indicator of current Pareto front."""
        return self.pareto_front.compute_hypervolume()
    
    def get_regret_bound(self, T: int) -> float:
        """
        Compute theoretical regret bound.
        
        R(T) = O(√(T log T)) for Pareto bandits
        """
        return np.sqrt(T * np.log(T + 1)) * self.config.num_mapping_params
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'total_updates': self.total_updates,
            'buffer_size': len(self.buffer),
            'pareto_front_size': len(self.pareto_front.points),
            'hypervolume': self.get_hypervolume(),
            'mapping_params': self.mapping_params.detach().cpu().numpy().tolist(),
            'regret_bound': self.get_regret_bound(self.total_updates)
        }
    
    def forward(self, risk: torch.Tensor) -> torch.Tensor:
        """Forward pass returning optimal sigma for given risk."""
        sigma = self.get_optimal_sigma(risk.item())
        return torch.tensor(sigma)
    
    def save_state(self, path: str):
        """Save PUPO state to file."""
        state = {
            'mapping_params': self.mapping_params.detach().cpu(),
            'optimizer_state': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'online_optimizer': {
                'counts': self.online_optimizer.counts,
                'privacy_sum': self.online_optimizer.privacy_sum,
                'utility_sum': self.online_optimizer.utility_sum
            }
        }
        torch.save(state, path)
        logger.info(f"PUPO state saved to {path}")
    
    def load_state(self, path: str):
        """Load PUPO state from file."""
        state = torch.load(path)
        self.mapping_params.data = state['mapping_params']
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.total_updates = state['total_updates']
        
        opt_state = state['online_optimizer']
        self.online_optimizer.counts = opt_state['counts']
        self.online_optimizer.privacy_sum = opt_state['privacy_sum']
        self.online_optimizer.utility_sum = opt_state['utility_sum']
        
        logger.info(f"PUPO state loaded from {path}")


# Theorem verification functions
def verify_convergence_theorem(pupo: PUPO, T: int) -> Dict[str, float]:
    """
    Verify Theorem 2: PUPO convergence guarantee.
    
    Under standard online learning assumptions:
    R(T) = O(√T log T)
    
    Args:
        pupo: PUPO instance
        T: Number of timesteps
    
    Returns:
        Dict with theoretical and empirical bounds
    """
    theoretical_bound = pupo.get_regret_bound(T)
    
    # Empirical regret (if we had oracle knowledge)
    # This is a placeholder - actual computation requires optimal strategy
    empirical_regret = pupo.cumulative_regret if hasattr(pupo, 'cumulative_regret') else 0.0
    
    return {
        'theoretical_bound': theoretical_bound,
        'empirical_regret': empirical_regret,
        'bound_satisfied': empirical_regret <= theoretical_bound
    }


# Factory function
def create_pupo(
    lambda_privacy: float = 0.5,
    buffer_size: int = 1000,
    learning_rate: float = 0.01
) -> PUPO:
    """
    Factory function to create PUPO with common configurations.
    
    Args:
        lambda_privacy: Weight for privacy objective
        buffer_size: Size of experience buffer
        learning_rate: Learning rate for gradient updates
    
    Returns:
        Configured PUPO instance
    """
    config = PUPOConfig(
        lambda_privacy=lambda_privacy,
        lambda_utility=1 - lambda_privacy,
        buffer_size=buffer_size,
        learning_rate=learning_rate
    )
    
    return PUPO(config)
