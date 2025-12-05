from setuptools import setup, find_packages

setup(
    name="iseer",
    version="0.1.0",
    description="Iseer Architecture: A Novel Mamba × MoE Hybrid Language Model",
    author="Inan",
    author_email="inan@iseer.co",
    url="https://github.com/InanXR/iseer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.1.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "train": [
            "wandb>=0.15.0",
            "datasets>=2.14.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
