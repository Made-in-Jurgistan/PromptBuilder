# Contributing to PromptBuilder

Thank you for considering contributing to PromptBuilder! This document outlines the process and guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors. By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/YOUR-USERNAME/promptbuilder.git
   cd promptbuilder
   ```
3. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/Made-in-Jurgistan/promptbuilder.git
   ```
4. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Environment

1. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Coding Standards

PromptBuilder follows these coding standards:

1. **PEP 8**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
2. **Type Hints**: Use type hints for function parameters and return values.
3. **Docstrings**: Use Google-style docstrings for all modules, classes, and functions.
4. **Code Formatting**: We use [Black](https://black.readthedocs.io/) for code formatting.
5. **Import Order**: We use [isort](https://pycqa.github.io/isort/) for consistent import ordering.
6. **Line Length**: Maximum line length is 88 characters (Black default).

Example of proper docstring format:

```python
def example_function(param1: str, param2: int) -> bool:
    """Short description of the function.
    
    More detailed explanation if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: Description of when this error is raised.
    """
    # Function implementation
    return True
```

## Testing Guidelines

1. **Write Tests**: All new features should include tests.
2. **Test Coverage**: Aim for at least 80% test coverage for new code.
3. **Run Tests**: Make sure all tests pass before submitting a PR.
   ```bash
   pytest
   ```
4. **Test Types**:
   - Unit tests for individual functions and classes
   - Integration tests for component interactions
   - End-to-end tests for complex workflows

## Pull Request Process

1. **Update Your Fork**: Before creating a PR, update your fork with the latest upstream changes:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git checkout your-feature-branch
   git merge main
   ```

2. **Create a Pull Request**: Push your changes to your fork and create a pull request from the GitHub interface.

3. **PR Description**: Include a clear description of the changes, the motivation for the changes, and any additional context.

4. **Continuous Integration**: Make sure all CI checks pass.

5. **Code Review**: Address any feedback from code reviews.

6. **Approval**: PRs require approval from at least one maintainer before merging.

7. **Merge**: Once approved, maintainers will merge the PR.

## Issue Reporting

When reporting issues, please include:

1. **Clear Title**: A descriptive title that summarizes the issue.
2. **Steps to Reproduce**: Detailed steps to reproduce the issue.
3. **Expected Behavior**: What you expected to happen.
4. **Actual Behavior**: What actually happened.
5. **Environment**: Information about your environment:
   - PromptBuilder version
   - Python version
   - Operating system
   - Any relevant dependencies

## Feature Requests

Feature requests are welcome! When suggesting a feature:

1. **Explain the Problem**: What problem does the feature solve?
2. **Propose a Solution**: How would the feature work?
3. **Describe Alternatives**: What alternatives have you considered?
4. **Additional Context**: Any other context or screenshots.

## Documentation

Clear documentation is critical for the project's usability:

1. **Code Documentation**: Ensure all code has proper docstrings.
2. **README Updates**: Update README.md for user-facing changes.
3. **Wiki Updates**: Update the wiki for detailed guides and examples.
4. **Documentation Tests**: Ensure examples in documentation work correctly.

## Release Process

The release process is managed by the maintainers:

1. **Version Bumping**: Version numbers follow [Semantic Versioning](https://semver.org/).
2. **Changelog**: All notable changes are documented in CHANGELOG.md.
3. **Release Notes**: Detailed release notes are created for each release.

---

Thank you for contributing to PromptBuilder! Your efforts help make the project better for everyone.

Made with ❤️ by Made in Jurgistan 