#!/usr/bin/env python
"""Setup script for Logan - Log Analysis Tool."""

import os
from setuptools import setup, find_packages
from logan import __version__


def read_requirements(filename="requirements.txt"):
    """Read dependencies from requirements.txt file."""
    requirements = []
    req_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    # Handle lines with environment markers
                    requirements.append(line)
    return requirements


def read_long_description():
    """Read long description from README.md."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="logan",
    version=__version__,
    description="Logan - Log Analysis Tool for preprocessing, templatization, and anomaly detection",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Logan Team",
    license="Apache-2.0",
    python_requires=">=3.9",
    keywords=[
        "log analysis",
        "anomaly detection",
        "log parsing",
        "drain3",
        "machine learning",
        "devops",
        "sre",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: System :: Logging",
        "Topic :: System :: Monitoring",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(
        where=".",
        include=["logan*"],
        exclude=["tests*", "tmp*", "examples*", "docs*"],
    ),
    package_data={
        "logan": [
            "drain/*.ini",
            "preprocessing/*.ini",
            "preprocessing/*.json",
            "log_diagnosis/templates/*.html",
            "log_diagnosis/templates/libs/*",
            "log_diagnosis/templates/static/*",
        ],
    },
    include_package_data=True,
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "logan=logan.cli:main",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/your-org/logan",
        "Documentation": "https://github.com/your-org/logan#readme",
        "Repository": "https://github.com/your-org/logan",
        "Issues": "https://github.com/your-org/logan/issues",
    },
)

