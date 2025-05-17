# 🚀 PromptBuilder

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-yellow.svg)

**A sophisticated query handling training data generator for fine-tuning LLMs as coding assistants**

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Command Line Interface](#-command-line-interface)
- [Usage Examples](#-usage-examples)
- [Architecture](#-architecture)
- [LLM Training Applications](#-llm-training-applications)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Overview

**PromptBuilder** provides a sophisticated system for generating high-quality training data for fine-tuning Large Language Models (LLMs) on query handling and execution tasks as coding assistants. It combines multiple reasoning frameworks, domain-specific knowledge, and quality assurance mechanisms to create training examples that promote comprehensive, error-resistant responses.

This system enables data scientists and ML engineers to generate diverse, realistic training examples across multiple programming domains, languages, and frameworks with minimal effort. The generated examples help LLMs learn how to:

- Explain complex technical concepts clearly
- Debug problematic code with systematic reasoning
- Solve coding problems step-by-step
- Optimize performance bottlenecks
- Respond to strategic decision questions
- Handle vague queries with clarification
- And much more!

## ✨ Key Features

- **🧩 Comprehensive reasoning frameworks** for different query types
- **🔬 Domain-specific code and technology knowledge** integration
- **📊 Multi-level difficulty scaling** from beginner to expert
- **💬 Dialogue simulation** with clarification and information gathering
- **📝 Production-quality code examples** with proper documentation and error handling
- **🔍 Advanced query understanding** with misinterpretation prevention
- **🧠 Detailed internal reasoning process** documentation
- **🛠️ Extensible architecture** for adding new domains and query types
- **📈 Comprehensive validation framework** to ensure quality

## 📥 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from PyPI

```bash
pip install promptbuilder
```

### Install from source

```bash
git clone https://github.com/Made-in-Jurgistan/promptbuilder.git
cd promptbuilder

# Install it
pip install -e .
```

## 🚀 Quick Start

```bash
# Should show the version number
python -m promptbuilder.main --version
```

## 🚀 How to Use

### Simple Command Line

```bash
# Create 5 examples and save them to a file
python -m promptbuilder.main generate --examples 5 --output examples.jsonl

# Answer questions about settings with a wizard
python -m promptbuilder.main
```

### Using in Your Python Code

```python
from promptbuilder.core.example_generator import ExampleGenerator
from promptbuilder.config import Config

# Set up your preferences
config = Config(
    num_examples_per_category=2,  # How many examples to create
    output_file="examples.jsonl"   # Where to save them
)

# Create the generator
generator = ExampleGenerator(config)

# Make the examples
examples = generator.generate_examples()

# See what you got
print(f"Created {len(examples)} training examples!")
```

## ⚙️ Settings You Can Change

You can customize how PromptBuilder works by changing these settings:

| Setting | What It Does | Default Value |
|---------|--------------|---------------|
| `num_examples_per_category` | How many examples to create for each type | 5 |
| `difficulty_levels` | Difficulty of the examples | ["basic", "intermediate", "advanced"] |
| `query_types` | Types of questions to generate | All types |
| `output_format` | File format to save examples | "jsonl" |
| `validate_output` | Check examples for quality | False |
| `use_parallel_processing` | Run faster using multiple cores | False |
| `selected_domains` | Which programming areas to include | [] (all domains) |

### Sample Settings File

You can save your settings in a JSON file:

```json
{
  "num_examples_per_category": 10,
  "difficulty_levels": ["basic", "intermediate"],
  "query_types": ["conceptExplanation", "debugging", "problemSolving"],
  "output_format": "jsonl",
  "output_file": "training_examples.jsonl",
  "validate_output": true,
  "use_parallel_processing": true,
  "selected_domains": ["web_development", "data_science"]
}
```

## 📝 Command Examples

### Create Examples About Web Development and Data Science

```bash
python -m promptbuilder.main generate --domains web_development,data_science --examples 10
```

### Create and Check Examples for Quality

```bash
python -m promptbuilder.main generate --validate --output examples.jsonl
```

### Check Examples You Already Created

```bash
python -m promptbuilder.main validate --input examples.jsonl --output validated_examples.jsonl
```

### See What Programming Topics Are Available

```bash
python -m promptbuilder.main domains --verbose
```

## 🏗️ How It Works

PromptBuilder works in a series of steps:

1. **Choose Settings**: Decide what kinds of examples to create
2. **Generate Questions**: Create realistic coding questions
3. **Create Responses**: Generate high-quality answers
4. **Quality Check**: Make sure the examples are good
5. **Save Results**: Store examples in your chosen format

The main parts that make it work are:

- **ExampleGenerator**: Creates the training examples
- **ComponentGenerator**: Makes the questions
- **FrameworkRegistry**: Picks the right answer style
- **ResponseGenerator**: Creates the answers
- **ExampleValidator**: Checks for quality
- **TemplateManager**: Handles question templates

## 👥 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Made with ❤️ by Made in Jurgistan 
