# bayesian_hmm_pyro/setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pyro_bayesian_hmm",
    version="0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Bayesian Hidden Markov Model implementation using Pyro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyro-bayesian-hmm",
    packages=find_packages(),
    package_data={"src": ["*.py"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "pyro-ppl",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "numba",
        "cupy-cuda11x",  # Adjust this based on your CUDA version
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=2.12.0",
            "pytest-randomly>=3.10.0",
            "pytest-xdist>=2.5.0",
        ],
    },
)
