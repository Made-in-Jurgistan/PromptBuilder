#!/usr/bin/env python3
"""
Setup script for PromptBuilder.

PromptBuilder is a comprehensive training data generation system 
for fine-tuning LLMs as coding assistants.

Author: Made in Jurgistan
Version: 2.0.0
License: MIT
"""

import os
import re
from setuptools import setup, find_packages
from typing import List

# Get package directory
package_dir = os.path.abspath(os.path.dirname(__file__))

# Get version from the __init__.py file
version = "2.0.0"  # Default version
init_file = os.path.join(package_dir, 'promptbuilder', '__init__.py')
if os.path.exists(init_file):
    with open(init_file, 'r', encoding='utf-8') as f:
        init_content = f.read()
        version_match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', init_content, re.MULTILINE)
        if version_match:
            version = version_match.group(1)

# Read long description from README.md
readme_path = os.path.join(package_dir, 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "PromptBuilder: Training data generator for fine-tuning LLMs as coding assistants"

def read_requirements(filename: str) -> List[str]:
    """Read requirements from file, skipping comments and empty lines."""
    requirements = []
    req_path = os.path.join(package_dir, filename)
    if not os.path.exists(req_path):
        return requirements
    with open(req_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

# Get requirements
install_requires = read_requirements('requirements.txt')
dev_requires = read_requirements('requirements-dev.txt') if os.path.exists(os.path.join(package_dir, 'requirements-dev.txt')) else []
doc_requires = read_requirements('requirements-docs.txt') if os.path.exists(os.path.join(package_dir, 'requirements-docs.txt')) else []

# If requirements list is empty, add core dependencies manually
if not install_requires:
    install_requires = [
        "pyyaml>=6.0",
        "jsonschema>=4.17.0",
        "tqdm>=4.64.0",
        "rich>=12.5.0",
    ]

setup(
    name="promptbuilder",
    version=version,
    author="Made in Jurgistan",
    author_email="madeinjurgistan@gmail.com",
    description="Query handling training data generator for fine-tuning LLMs as coding assistants",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Made-in-Jurgistan/promptbuilder",
    project_urls={
        "Bug Tracker": "https://github.com/Made-in-Jurgistan/promptbuilder/issues",
        "Documentation": "https://promptbuilder.readthedocs.io/",
        "Source Code": "https://github.com/Made-in-Jurgistan/promptbuilder",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={
        "promptbuilder": [
            "templates/*.json",
            "schemas/*.json",
            "data/*.yaml",
            "py.typed"
        ],
    },
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": doc_requires,
        "all": dev_requires + doc_requires,
    },
    entry_points={
        "console_scripts": [
            "promptbuilder=promptbuilder.main:main"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    keywords="llm, prompt engineering, training data, code assistant, nlp, ai, machine learning",
    zip_safe=False,
)
