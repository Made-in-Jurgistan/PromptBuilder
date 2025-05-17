# 🚀 PromptBuilder

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/Made-in-Jurgistan/promptbuilder/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-yellow.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A comprehensive framework for generating high-quality training data for fine-tuning LLMs as coding assistants.**

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [CLI Interface](#cli-interface)
  - [Python API](#python-api)
  - [Docker Deployment](#docker-deployment)
- [Configuration Reference](#configuration-reference)
- [Extending PromptBuilder](#extending-promptbuilder)
- [Performance Considerations](#performance-considerations)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

PromptBuilder is a sophisticated framework designed to generate diverse, high-quality training examples for fine-tuning Large Language Models (LLMs) for code-related tasks. By combining advanced reasoning frameworks, domain-specific knowledge representation, and comprehensive quality assurance mechanisms, it produces training data that helps LLMs develop robust, accurate coding assistance capabilities.

The system addresses a critical challenge in LLM training: creating realistic, diverse examples that capture the nuances of different programming domains, reasoning approaches, and difficulty levels. This enables fine-tuned models to provide more accurate, contextually appropriate coding assistance.

## ✨ Key Features

- **Multi-framework Reasoning**: Implements distinct reasoning approaches for different query types:
  - Developer Clarification Model for concept explanations
  - Chain of Thought Framework for systematic problem solving
  - Hybrid Decision-Making Framework for evaluating options

- **Comprehensive Domain Coverage**:
  - Multiple programming languages (Python, JavaScript, TypeScript, R)
  - Domain-specific libraries and frameworks
  - Language-appropriate code snippets and error patterns

- **Granular Difficulty Scaling**:
  - Basic: Entry-level explanations and problems
  - Intermediate: Domain-specific applications with moderate complexity
  - Advanced: Production-level challenges with multiple considerations

- **Quality Assurance Pipeline**:
  - Automatic validation of generated examples
  - Configurable quality thresholds and criteria
  - Metrics collection for training data quality

- **Modular, Extensible Architecture**:
  - Clean separation between components
  - Registry pattern for frameworks and domains
  - Plugin architecture for adding new reasoning patterns

## 🏗️ Architecture

PromptBuilder follows a pipeline architecture with these key components:

```
┌─────────────────┐     ┌──────────────────┐       ┌───────────────────┐
│ Configuration   │───▶│Example Generator  │───▶  │Component Generator│
└─────────────────┘     └──────────────────┘       └───────────────────┘
                                │                        │
                                ▼                        │
                      ┌──────────────────┐               │
                      │ Template Manager │◀─────────────┘
                      └──────────────────┘
                                │
                                ▼
                      ┌──────────────────┐     ┌───────────────────┐
                      │ Response Generator│───▶│ Example Validator │
                      └──────────────────┘     └───────────────────┘
                                │                        │
                                ▼                        ▼
                      ┌──────────────────┐    ┌───────────────────┐
                      │ Output Formatter │    │ Metrics Collection│
                      └──────────────────┘    └───────────────────┘
```

Each component is designed with clear interfaces and separation of concerns, enabling:
1. Independent testing and validation
2. Extension with new query types and frameworks
3. Customization of generation parameters
4. Structured patterns for response creation

## 📥 Installation

### From PyPI (coming soon)

```bash
pip install promptbuilder
```

### From Source

```bash
# Clone the repository
git clone https://github.com/Made-in-Jurgistan/promptbuilder.git
cd promptbuilder

# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies and development version
pip install -e ".[dev]"  # Include development dependencies
```

### Requirements

- **Python**: 3.8 or higher
- **Dependencies**: See `requirements.txt` for production dependencies
- **Development**: Additional dependencies in `requirements-dev.txt`

## 📚 Usage Guide

### CLI Interface

The command-line interface provides comprehensive controls for generating, validating, and analyzing examples:

```bash
# Generate 50 examples with specific parameters
promptbuilder generate \
  --examples 50 \
  --domains web_development,data_science \
  --difficulties basic,intermediate \
  --output training_data.jsonl \
  --validate \
  --seed 42

# Validate existing examples
promptbuilder validate \
  --input examples.jsonl \
  --threshold 0.85 \
  --output validated_examples.jsonl

# Display domain information
promptbuilder domains --verbose

# Get framework information
promptbuilder frameworks --verbose

# Show available options
promptbuilder --help
```

Full CLI reference:

```
Usage: promptbuilder [OPTIONS] [COMMAND]

Commands:
  generate     Generate training examples
  validate     Validate existing examples
  stats        Display statistics for generated examples
  domains      List available domains
  frameworks   List available reasoning frameworks
  version      Display version information

Options:
  --config PATH                Path to configuration file
  --output PATH                Output file path [default: training_data.jsonl]
  --format {jsonl,csv,md}      Output format [default: jsonl]
  --examples INT               Examples per category [default: 5]
  --domains TEXT               Comma-separated domains to include
  --difficulties TEXT          Comma-separated difficulty levels
  --query-types TEXT           Comma-separated query types
  --seed INT                   Random seed for reproducibility [default: 42]
  --validate / --no-validate   Validate generated examples [default: False]
  --parallel / --no-parallel   Use parallel processing [default: False]
  --verbose / --no-verbose     Enable verbose output [default: False]
  --log-file PATH              Log file path [default: None]
  --help                       Show this message and exit
```

### Python API

The Python API provides programmatic access to all functionality with fine-grained control:

```python
from promptbuilder.core.example_generator import ExampleGenerator
from promptbuilder.config import Config
from promptbuilder.core.example_validator import ExampleValidator

# Create configuration with custom settings
config = Config(
    num_examples_per_category=10,
    selected_domains=["web_development", "data_science"],
    difficulty_levels=["intermediate", "advanced"],
    query_types=["conceptExplanation", "debugging", "optimization"],
    output_format="jsonl",
    validate_output=True,
    use_parallel_processing=True
)

# Initialize generator
generator = ExampleGenerator(config)

# Generate examples
examples = generator.generate_examples()
print(f"Generated {len(examples)} examples")

# Validate examples with custom settings
validator = ExampleValidator(
    supported_languages=generator.supported_languages,
    supported_technologies=generator.supported_technologies,
    config={"threshold": 0.85, "strict": True}
)
valid_examples, metrics = validator.validate_examples(examples)
print(f"Validation passed: {len(valid_examples)}/{len(examples)}")

# Export to file
with open("high_quality_examples.jsonl", "w") as f:
    for example in valid_examples:
        f.write(example.to_json() + "\n")
```

### 🐳 Docker Deployment

For containerized environments, use the provided Docker configuration:

```bash
# Build the container image
docker build -t promptbuilder:2.0.0 .

# Run with volume mount for output
docker run -v "$(pwd)/output:/data" promptbuilder:2.0.0

# With custom configuration (on Windows)
docker run -v "%cd%\output:/data" -v "%cd%\config.json:/app/config.json" \
  promptbuilder:2.0.0 generate --config /app/config.json --examples 100
```

## ⚙️ Configuration Reference

Configuration can be provided via JSON file or programmatically. Example JSON configuration:

```json
{
  "num_examples_per_category": 10,
  "difficulty_levels": ["basic", "intermediate", "advanced"],
  "query_types": ["conceptExplanation", "debugging", "optimization"],
  "output_format": "jsonl",
  "output_file": "examples.jsonl",
  "validate_output": true,
  "validation_threshold": 0.85,
  "use_parallel_processing": true,
  "selected_domains": ["web_development", "data_science"],
  "random_seed": 42,
  "logging": {
    "level": "INFO",
    "file": "generation.log"
  }
}
```

## 🧩 Extending PromptBuilder

PromptBuilder's modular design allows for straightforward extensions:

### Adding New Domains

Create a domain module in `promptbuilder/domains/` following the template pattern:

```python
# promptbuilder/domains/blockchain.py
"""Domain knowledge for blockchain development."""

DOMAIN_METADATA = {
    "name": "blockchain",
    "description": "Blockchain development and smart contracts",
    "languages": ["Solidity", "Rust", "JavaScript"],
    "technologies": [
        {
            "name": "Ethereum",
            "description": "Smart contract platform with EVM",
            "version_range": ["1.0.0", "2.0.0"]
        },
        # Additional technologies...
    ],
    "concepts": [
        {
            "name": "Gas optimization",
            "difficulty": "advanced",
            "related": ["transaction fees", "computational efficiency"]
        },
        # Additional concepts...
    ]
}
```

### Creating Custom Reasoning Frameworks

Implement a new framework by extending the base classes:

```python
from promptbuilder.core.reasoning import Framework, ReasoningStep

class MyCustomFramework(Framework):
    """Custom reasoning framework for specialized domains."""
    
    name = "Custom Domain-Specific Reasoning"
    
    def generate_reasoning_steps(self, query_info):
        """Generate reasoning steps for this framework."""
        return [
            ReasoningStep(
                title="Domain Analysis",
                content=f"Analyzing the {query_info.domain} domain context..."
            ),
            # Additional steps...
        ]
```

## ⚡ Performance Considerations

- **Memory Usage**: Example generation with large datasets may require 4GB+ RAM
- **Parallelization**: Enable with `--parallel` for multi-core systems (~2-4x speedup)
- **Storage**: Generated datasets typically require ~1MB per 100 examples
- **Processing Time**: ~5-10 examples per second on modern hardware

## 📘 API Reference

See the [API Documentation](https://promptbuilder.readthedocs.io/) for detailed reference on all public APIs.

Key modules:

- **promptbuilder.core.example_generator**: Primary interface for example generation
- **promptbuilder.core.example_validator**: Validation and quality assessment
- **promptbuilder.core.query_components**: Data structures for query representation
- **promptbuilder.config**: Configuration management

## 👥 Contributing

We welcome contributions to PromptBuilder! 

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by Made in Jurgistan
