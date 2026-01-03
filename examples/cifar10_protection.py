"""
RAPTOR Example: Protecting a CIFAR-10 Classifier

This example demonstrates how to use RAPTOR to protect a trained
image classifier against privacy attacks at inference time.

Author: H M Shujaat Zaheer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import RAPTOR
from raptor import RAPTORFramework, RAPTORConfig, create_raptor


# Simple CNN for demonstration
class SimpleCNN(nn.Module):
    """Simple CNN classifier for CIFAR-10."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def simulate_membership_inference_attack(
    model: nn.Module,
    raptor: RAPTORFramework,
    member_data: torch.Tensor,
    non_member_data: torch.Tensor,
    num_samples: int = 100
) -> dict:
    """
    Simulate membership inference attack.
    
    Compares attack success with and without RAPTOR protection.
    """
    model.eval()
    
    results = {
        'without_raptor': {'member_conf': [], 'non_member_conf': []},
        'with_raptor': {'member_conf': [], 'non_member_conf': []}
    }
    
    # Sample data
    member_indices = np.random.choice(len(member_data), num_samples, replace=False)
    non_member_indices = np.random.choice(len(non_member_data), num_samples, replace=False)
    
    with torch.no_grad():
        # Test on members
        for idx in member_indices:
            x = member_data[idx:idx+1]
            
            # Without RAPTOR
            output = model(x)
            conf = F.softmax(output, dim=-1).max().item()
            results['without_raptor']['member_conf'].append(conf)
            
            # With RAPTOR
            protected_output = raptor.process(x, output)
            protected_conf = F.softmax(protected_output, dim=-1).max().item()
            results['with_raptor']['member_conf'].append(protected_conf)
        
        # Test on non-members
        for idx in non_member_indices:
            x = non_member_data[idx:idx+1]
            
            # Without RAPTOR
            output = model(x)
            conf = F.softmax(output, dim=-1).max().item()
            results['without_raptor']['non_member_conf'].append(conf)
            
            # With RAPTOR
            protected_output = raptor.process(x, output)
            protected_conf = F.softmax(protected_output, dim=-1).max().item()
            results['with_raptor']['non_member_conf'].append(protected_conf)
    
    # Compute attack accuracy (threshold-based)
    def compute_attack_accuracy(member_confs, non_member_confs):
        threshold = np.median(member_confs + non_member_confs)
        member_correct = sum(1 for c in member_confs if c > threshold)
        non_member_correct = sum(1 for c in non_member_confs if c <= threshold)
        return (member_correct + non_member_correct) / (len(member_confs) + len(non_member_confs))
    
    results['without_raptor']['attack_accuracy'] = compute_attack_accuracy(
        results['without_raptor']['member_conf'],
        results['without_raptor']['non_member_conf']
    )
    
    results['with_raptor']['attack_accuracy'] = compute_attack_accuracy(
        results['with_raptor']['member_conf'],
        results['with_raptor']['non_member_conf']
    )
    
    return results


def main():
    """Main example demonstrating RAPTOR usage."""
    
    print("=" * 60)
    print("RAPTOR: Real-time Adaptive Privacy Through Output Regulation")
    print("=" * 60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\n1. Creating model...")
    model = SimpleCNN(num_classes=10).to(device)
    
    # Initialize with random weights (in practice, load trained model)
    # model.load_state_dict(torch.load('cifar10_model.pth'))
    
    # Create RAPTOR framework
    print("\n2. Initializing RAPTOR...")
    raptor = create_raptor(
        model=model,
        sigma_range=(0.01, 0.5),
        lambda_privacy=0.6
    )
    
    # Simulate data
    print("\n3. Generating synthetic data...")
    member_data = torch.randn(500, 3, 32, 32).to(device)
    non_member_data = torch.randn(500, 3, 32, 32).to(device)
    
    # Calibrate RPLQ with member/non-member data
    print("\n4. Calibrating RAPTOR...")
    raptor.calibrate(
        member_samples=member_data[:100],
        non_member_samples=non_member_data[:100]
    )
    
    # Example inference
    print("\n5. Running protected inference...")
    test_query = torch.randn(1, 3, 32, 32).to(device)
    
    # Manual usage
    protected_output, metrics = raptor.process(test_query, return_metrics=True)
    
    print(f"\n   Privacy Risk (ρ): {metrics['rho']:.4f}")
    print(f"   - MIA Risk (φ_MI): {metrics['phi_mi']:.4f}")
    print(f"   - Inversion Risk (φ_Inv): {metrics['phi_inv']:.4f}")
    print(f"   - Extraction Risk (φ_Ext): {metrics['phi_ext']:.4f}")
    print(f"   Applied Noise (σ): {metrics['sigma']:.4f}")
    print(f"   Privacy Budget (ε): {metrics['epsilon']:.4f}")
    print(f"   Latency: {metrics['latency_ms']:.2f} ms")
    
    # Using decorator
    print("\n6. Using @raptor.protect decorator...")
    
    @raptor.protect
    def secure_predict(x):
        return model(x)
    
    protected_pred = secure_predict(test_query)
    print(f"   Protected prediction shape: {protected_pred.shape}")
    
    # Run attack simulation
    print("\n7. Simulating membership inference attack...")
    attack_results = simulate_membership_inference_attack(
        model, raptor, member_data, non_member_data, num_samples=100
    )
    
    print(f"\n   Attack Results:")
    print(f"   - Without RAPTOR: {attack_results['without_raptor']['attack_accuracy']:.2%} accuracy")
    print(f"   - With RAPTOR: {attack_results['with_raptor']['attack_accuracy']:.2%} accuracy")
    print(f"   - Protection Improvement: {(attack_results['without_raptor']['attack_accuracy'] - attack_results['with_raptor']['attack_accuracy']):.2%}")
    
    # Process multiple queries to build Pareto front
    print("\n8. Processing queries to learn Pareto front...")
    for i in tqdm(range(200), desc="   Processing"):
        query = torch.randn(1, 3, 32, 32).to(device)
        _ = raptor.process(query)
    
    # Get statistics
    print("\n9. Framework Statistics:")
    stats = raptor.get_statistics()
    print(f"   - Total queries: {stats['total_queries']}")
    print(f"   - Pareto front size: {stats['pupo']['pareto_front_size']}")
    print(f"   - Hypervolume: {stats['pupo']['hypervolume']:.4f}")
    
    if 'query_stats' in stats:
        print(f"   - Mean risk: {stats['query_stats']['mean_risk']:.4f}")
        print(f"   - Mean latency: {stats['query_stats']['mean_latency_ms']:.2f} ms")
    
    # Privacy guarantee
    print("\n10. Privacy Guarantee (Theorem 1):")
    guarantee = raptor.get_privacy_guarantee()
    print(f"   - Total ε: {guarantee['epsilon_total']:.4f}")
    print(f"   - Max per-query ε: {guarantee['max_per_query_epsilon']:.4f}")
    print(f"   - Queries processed: {guarantee['num_queries']}")
    print(f"   - δ: {guarantee['delta']}")
    
    print("\n" + "=" * 60)
    print("RAPTOR demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
