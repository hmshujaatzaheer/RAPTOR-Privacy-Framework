"""
RAPTOR: Real-time Adaptive Privacy Through Output Regulation

Setup script for package installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="raptor-privacy",
    version="0.1.0",
    author="H M Shujaat Zaheer",
    author_email="shujabis@gmail.com",
    description="Real-time Adaptive Privacy Through Output Regulation for ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hmshujaatzaheer/RAPTOR-Privacy-Framework",
    project_urls={
        "Bug Tracker": "https://github.com/hmshujaatzaheer/RAPTOR-Privacy-Framework/issues",
        "Documentation": "https://github.com/hmshujaatzaheer/RAPTOR-Privacy-Framework/docs",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "full": requirements,
    },
    entry_points={
        "console_scripts": [
            "raptor-audit=raptor.cli:audit_model",
        ],
    },
)
