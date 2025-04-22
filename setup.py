from setuptools import setup, find_packages

setup(
    name="economy_predictor",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "qdrant-client>=1.0.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ]
    },
)