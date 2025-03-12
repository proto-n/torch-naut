import torch
import torch.nn as nn
import pytest
from torchnaut.crps import EpsilonSampler

def test_n_samples_context():
    sampler = EpsilonSampler(2)
    x = torch.randn(4, 3)  # batch_size=4, features=3
    
    # Default behavior
    out = sampler(x)
    assert out.shape == (4, 100, 5)  # default n_samples=100
    
    # Within context
    with EpsilonSampler.n_samples(50):
        out = sampler(x)
        assert out.shape == (4, 50, 5)
    
    # After context - should return to default
    out = sampler(x)
    assert out.shape == (4, 100, 5)

def test_nested_n_samples_context():
    sampler = EpsilonSampler(2)
    x = torch.randn(4, 3)
    
    with EpsilonSampler.n_samples(50):
        out1 = sampler(x)
        assert out1.shape == (4, 50, 5)
        
        with EpsilonSampler.n_samples(25):
            out2 = sampler(x)
            assert out2.shape == (4, 25, 5)
        
        out3 = sampler(x)
        assert out3.shape == (4, 50, 5)
    
    out4 = sampler(x)
    assert out4.shape == (4, 100, 5)

def test_n_samples_with_model():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(3, 4),
                EpsilonSampler(2),
                nn.Linear(6, 1)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    model = TestModel()
    x = torch.randn(4, 3)
    
    # Default behavior
    out = model(x)
    assert out.shape == (4, 100, 1)
    
    # With context
    with EpsilonSampler.n_samples(30):
        out = model(x)
        assert out.shape == (4, 30, 1)
    
    # After context
    out = model(x)
    assert out.shape == (4, 100, 1)
