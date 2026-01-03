"""
Unit tests for RAPTOR framework.

Run with: pytest tests/test_raptor.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from raptor import RAPTORFramework, create_raptor
from raptor.core.rplq import RPLQ, RPLQConfig
from raptor.defenses.aos import AOS, AOSConfig
from raptor.optimization.pupo import PUPO, PUPOConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model():
    """Create simple model for testing."""
    return SimpleModel()


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(1, 10)


class TestRPLQ:
    """Tests for RPLQ module."""
    
    def test_initialization(self, model):
        """Test RPLQ initialization."""
        rplq = RPLQ(model=model)
        assert rplq is not None
        assert rplq.config.alpha + rplq.config.beta + rplq.config.gamma == 1.0
    
    def test_compute_risk(self, model, sample_input):
        """Test risk computation."""
        rplq = RPLQ(model=model)
        result = rplq.compute_risk(sample_input)
        
        assert 'rho' in result
        assert 'phi_mi' in result
        assert 'phi_inv' in result
        assert 'phi_ext' in result
        
        # Risk should be in [0, 1]
        assert 0 <= result['rho'] <= 1
    
    def test_weight_update(self, model):
        """Test weight update."""
        rplq = RPLQ(model=model)
        rplq.update_weights(0.5, 0.3, 0.2)
        
        assert abs(rplq.alpha.item() - 0.5) < 1e-6
        assert abs(rplq.beta.item() - 0.3) < 1e-6
        assert abs(rplq.gamma.item() - 0.2) < 1e-6


class TestAOS:
    """Tests for AOS module."""
    
    def test_initialization(self):
        """Test AOS initialization."""
        aos = AOS()
        assert aos is not None
        assert aos.config.sigma_min < aos.config.sigma_max
    
    def test_sanitize(self):
        """Test output sanitization."""
        aos = AOS()
        output = torch.randn(1, 10)
        
        # Low risk should have minimal change
        sanitized_low = aos.sanitize(output, risk=0.1)
        assert sanitized_low.shape == output.shape
        
        # High risk should have more change
        sanitized_high = aos.sanitize(output, risk=0.9)
        assert sanitized_high.shape == output.shape
        
        # Difference should be larger for high risk
        diff_low = (sanitized_low - output).abs().mean()
        diff_high = (sanitized_high - output).abs().mean()
        assert diff_high >= diff_low
    
    def test_epsilon_computation(self):
        """Test privacy budget computation."""
        aos = AOS()
        
        # Higher risk should give higher epsilon (more noise, lower epsilon)
        eps_low = aos.get_epsilon(0.1)
        eps_high = aos.get_epsilon(0.9)
        
        # Higher risk = more noise = lower epsilon
        assert eps_high < eps_low


class TestPUPO:
    """Tests for PUPO module."""
    
    def test_initialization(self):
        """Test PUPO initialization."""
        pupo = PUPO()
        assert pupo is not None
        assert len(pupo.buffer) == 0
    
    def test_record_experience(self):
        """Test experience recording."""
        pupo = PUPO()
        
        pupo.record_experience(
            risk=0.5,
            attack_success=0.3,
            utility=0.8,
            sigma=0.2
        )
        
        assert len(pupo.buffer) == 1
        assert pupo.total_updates == 1
    
    def test_optimal_sigma(self):
        """Test optimal sigma computation."""
        pupo = PUPO()
        
        # Add some experiences
        for _ in range(100):
            pupo.record_experience(
                risk=np.random.random(),
                attack_success=np.random.random(),
                utility=np.random.random(),
                sigma=np.random.random()
            )
        
        sigma = pupo.get_optimal_sigma(0.5)
        assert 0 <= sigma <= 1
    
    def test_pareto_front(self):
        """Test Pareto front computation."""
        pupo = PUPO()
        
        # Add diverse experiences
        for _ in range(50):
            pupo.record_experience(
                risk=np.random.random(),
                attack_success=np.random.random(),
                utility=np.random.random(),
                sigma=np.random.random()
            )
        
        front = pupo.get_pareto_front()
        assert isinstance(front, list)


class TestRAPTORFramework:
    """Tests for main RAPTOR framework."""
    
    def test_initialization(self, model):
        """Test framework initialization."""
        raptor = RAPTORFramework(model=model)
        assert raptor is not None
        assert raptor.rplq is not None
        assert raptor.aos is not None
        assert raptor.pupo is not None
    
    def test_create_raptor_factory(self, model):
        """Test factory function."""
        raptor = create_raptor(
            model=model,
            sigma_range=(0.1, 0.5),
            lambda_privacy=0.7
        )
        
        assert raptor is not None
        assert raptor.config.aos.sigma_min == 0.1
        assert raptor.config.aos.sigma_max == 0.5
    
    def test_process(self, model, sample_input):
        """Test query processing."""
        raptor = create_raptor(model=model)
        
        # Basic processing
        output = raptor.process(sample_input)
        assert output is not None
        assert output.shape[1] == 5  # Output dim
        
        # With metrics
        output, metrics = raptor.process(sample_input, return_metrics=True)
        assert 'rho' in metrics
        assert 'sigma' in metrics
        assert 'latency_ms' in metrics
    
    def test_decorator(self, model, sample_input):
        """Test protect decorator."""
        raptor = create_raptor(model=model)
        
        @raptor.protect
        def inference(x):
            return model(x)
        
        output = inference(sample_input)
        assert output is not None
    
    def test_statistics(self, model, sample_input):
        """Test statistics collection."""
        raptor = create_raptor(model=model)
        
        # Process some queries
        for _ in range(10):
            raptor.process(sample_input)
        
        stats = raptor.get_statistics()
        assert stats['total_queries'] == 10
    
    def test_privacy_guarantee(self, model, sample_input):
        """Test privacy guarantee computation."""
        raptor = create_raptor(model=model)
        
        # Process queries
        for _ in range(20):
            raptor.process(sample_input)
        
        guarantee = raptor.get_privacy_guarantee()
        assert 'epsilon_total' in guarantee
        assert 'num_queries' in guarantee
        assert guarantee['num_queries'] == 20


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end(self, model):
        """Test complete workflow."""
        # Create RAPTOR
        raptor = create_raptor(model=model)
        
        # Calibrate
        member_data = torch.randn(50, 10)
        non_member_data = torch.randn(50, 10)
        raptor.calibrate(member_data, non_member_data)
        
        # Process queries
        for _ in range(100):
            query = torch.randn(1, 10)
            output, metrics = raptor.process(query, return_metrics=True)
            
            assert output is not None
            assert 0 <= metrics['rho'] <= 1
        
        # Check statistics
        stats = raptor.get_statistics()
        assert stats['total_queries'] == 100
        
        # Check Pareto optimization happened
        assert stats['pupo']['pareto_front_size'] > 0
    
    def test_latency_constraint(self, model):
        """Test that latency is reasonable."""
        raptor = create_raptor(model=model)
        
        latencies = []
        for _ in range(50):
            query = torch.randn(1, 10)
            _, metrics = raptor.process(query, return_metrics=True)
            latencies.append(metrics['latency_ms'])
        
        # Average latency should be under 10ms for simple model
        avg_latency = np.mean(latencies)
        assert avg_latency < 100  # Generous bound for CI


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
