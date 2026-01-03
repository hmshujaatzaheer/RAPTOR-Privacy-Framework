"""
RAPTOR: Real-time Adaptive Privacy Through Output Regulation

A unified framework for privacy measurement and adaptive defense in ML systems.

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
GitHub: https://github.com/hmshujaatzaheer/RAPTOR-Privacy-Framework
"""

__version__ = "0.1.0"
__author__ = "H M Shujaat Zaheer"
__email__ = "shujabis@gmail.com"

from .core.framework import RAPTORFramework, RAPTORConfig, create_raptor
from .core.rplq import RPLQ, RPLQConfig, compute_privacy_risk
from .defenses.aos import AOS, AOSConfig, create_aos
from .optimization.pupo import PUPO, PUPOConfig, create_pupo

__all__ = [
    # Main framework
    'RAPTORFramework',
    'RAPTORConfig',
    'create_raptor',
    
    # RPLQ
    'RPLQ',
    'RPLQConfig',
    'compute_privacy_risk',
    
    # AOS
    'AOS',
    'AOSConfig',
    'create_aos',
    
    # PUPO
    'PUPO',
    'PUPOConfig',
    'create_pupo',
]
