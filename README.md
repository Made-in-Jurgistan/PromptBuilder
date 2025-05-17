# 🚀 PromptBuilder

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> A comprehensive training data generation system for fine-tuning LLMs as coding assistants.

## 📋 Overview

PromptBuilder is a powerful tool for generating high-quality training examples for LLMs. It combines multiple reasoning frameworks, domain-specific knowledge, and quality assurance processes to create diverse and realistic training examples.

## 🔧 Installation

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

# Install the package in development mode
pip install -e .
```

## 🚀 Quick Start

Once installed, you can use PromptBuilder with the following commands:

```bash
# Generate 10 examples
promptbuilder generate --examples 10 --output examples.jsonl

# List available domains
promptbuilder domains

# Show version information
promptbuilder version
```

## 🐳 Docker Deployment

For a containerized deployment, use the included Dockerfile:

```bash
# Build the Docker image
docker build -t promptbuilder:2.0.0 .

# Run the container with volume mount
docker run -v "$(pwd)/output:/data" promptbuilder:2.0.0

# For Windows Command Prompt
docker run -v "%cd%\output:/data" promptbuilder:2.0.0

# For Windows PowerShell
docker run -v "${PWD}/output:/data" promptbuilder:2.0.0
```

## 🛠️ Command Line Interface

PromptBuilder offers a comprehensive CLI:

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

## ✅ System Requirements

### Minimum Requirements

- **Python**: Version 3.8 or higher
- **CPU**: 2+ cores
- **RAM**: 4GB minimum
- **Storage**: 1GB free space
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+ or compatible Linux distribution

## 📊 Example

```python
from promptbuilder.core.example_generator import ExampleGenerator
from promptbuilder.config import Config

# Create configuration
config = Config(num_examples_per_category=5)
generator = ExampleGenerator(config)

# Generate examples
examples = generator.generate_examples()

# Save to file
with open("examples.jsonl", "w") as f:
    for example in examples:
        f.write(example.to_json() + "\n")
```

## 📮 Support

- **GitHub Issues**: [github.com/Made-in-Jurgistan/promptbuilder/issues](https://github.com/Made-in-Jurgistan/promptbuilder/issues)
- **Email**: [madeinjurgistan@gmail.com](mailto:madeinjurgistan@gmail.com)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 