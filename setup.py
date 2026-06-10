from setuptools import setup, find_packages

setup(
    name="dual_lora_math",
    version="0.1.0",
    description="Dual LoRA for Mathematical Reasoning with LLEMMA-7B",
    author="Saley",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch==2.6.0",
        "transformers==4.38.2",
        "peft==0.9.0",
        "accelerate==0.27.2",
        "datasets==2.17.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "wandb>=0.16.0",
        "sympy>=1.12",
        "bitsandbytes==0.43.0",
        "sentencepiece>=0.1.99",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
)