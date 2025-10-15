# Contributing to Quantum Transpilation Analysis

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Guidelines](#development-guidelines)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to professional and respectful collaboration standards. All contributors are expected to:
- Be respectful and inclusive in discussions
- Provide constructive feedback
- Focus on what is best for the community and research advancement
- Show empathy towards other community members

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Git for version control
- Basic understanding of quantum computing concepts
- Familiarity with Qiskit and quantum circuit transpilation

### Development Setup

1. Fork the repository on GitHub

2. Clone your fork locally:
   ```bash
   git clone https://github.com/ksupasate/SupermarQ-Noise-Adaptive.git
   cd SupermarQ-Noise-Adaptive
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies including development tools:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**: Found a bug? Please report it!
2. **Feature Requests**: Have an idea? We'd love to hear it!
3. **Code Contributions**:
   - Bug fixes
   - New quantum algorithms
   - Performance improvements
   - Documentation improvements
4. **Documentation**:
   - Improve existing docs
   - Add tutorials or examples
   - Fix typos or clarify explanations

### What We're Looking For

- **New Quantum Algorithms**: Implementations of additional algorithms for benchmarking
- **Enhanced Metrics**: New SupermarQ metrics or analysis techniques
- **Optimization Improvements**: Better transpilation strategies
- **Visualization Enhancements**: More insightful plots and data representations
- **Educational Content**: Tutorial notebooks and learning materials
- **Performance Optimizations**: Faster execution or reduced memory usage

## Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) Python style guidelines
- Use meaningful variable and function names
- Add type hints to all function signatures
- Maximum line length: 100 characters

### Documentation Standards

- **Docstrings**: Use Google-style docstrings for all functions and classes
  ```python
  def example_function(param1: int, param2: str) -> bool:
      """Brief description of function.

      Longer description providing more context about what the function
      does and how it should be used.

      Args:
          param1: Description of first parameter
          param2: Description of second parameter

      Returns:
          Description of return value

      Raises:
          ValueError: When invalid input is provided
      """
      pass
  ```

- **Comments**: Add explanatory comments for complex quantum operations
- **README**: Update README.md if adding new features or changing usage

### Quantum Circuit Guidelines

- Use Qiskit best practices for circuit construction
- Include circuit diagrams or visualizations for complex circuits
- Provide references to quantum algorithms from academic papers
- Document quantum gate choices and their purpose

### Testing

- Test your code with the provided algorithms
- Verify outputs match expected behavior
- Run the full experiment pipeline to ensure no regressions
- Test on Python 3.9, 3.10, and 3.11 if possible

### Commit Messages

Write clear, descriptive commit messages:

```
Add Quantum Approximate Optimization Algorithm (QAOA)

- Implement QAOA circuit for MaxCut problem
- Add transpilation benchmarking for QAOA
- Update documentation with QAOA description
- Add unit tests for circuit generation
```

Format:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed explanation if needed
- List specific changes with bullet points

## Submitting Changes

### Pull Request Process

1. **Update Documentation**: Ensure README, docstrings, and comments are current

2. **Self-Review**: Review your own code for:
   - Code quality and style
   - Proper error handling
   - Comprehensive docstrings
   - No debugging print statements

3. **Test Thoroughly**: Run all experiments and verify outputs

4. **Create Pull Request**:
   - Provide a clear title describing the change
   - Reference any related issues (e.g., "Fixes #123")
   - Describe what changed and why
   - Include screenshots for visualization changes
   - List any breaking changes

5. **Address Feedback**: Respond to review comments promptly

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Changes Made
- List specific changes
- With bullet points

## Testing
Describe testing performed

## Screenshots (if applicable)
Add visualizations or output comparisons

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] All tests pass
- [ ] No new warnings introduced
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps to reproduce the behavior
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - Python version
   - Qiskit version
   - Operating system
   - Output of `pip freeze`
6. **Error Messages**: Full error traceback if applicable
7. **Code Sample**: Minimal code that reproduces the issue

### Feature Requests

For feature requests, please include:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How would you implement this?
3. **Alternatives Considered**: Other approaches you've thought about
4. **Additional Context**: Any relevant research papers or references

## Questions?

If you have questions about contributing:
- Open an issue with the "question" label
- Refer to the [README.md](README.md) for project overview
- Check the [tutorial notebook](tutorial.ipynb) for examples

## Attribution

Contributors will be acknowledged in the project documentation. By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to advancing quantum computing research!
