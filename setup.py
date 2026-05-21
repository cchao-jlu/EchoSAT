from setuptools import find_packages, setup

setup(
    name="echosat",
    packages=find_packages(),
    install_requires=[
        "torch==2.5.0",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "triton==3.1.0",
        "joblib==1.4.2",
        "torch_geometric==2.4.0",
        "pandas==2.2.2",
        "hydra-core",
        "wandb",
        "tqdm",
        "seaborn",
        "python-sat",
        "cnfgen",
        "jupyter",
        "torchmetrics",
    ],
    version="0.0.1",
)
