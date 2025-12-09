# Contributing to Jailbreak Harness

## 🤝 How to Contribute

We welcome contributions! Areas of interest:

- New test categories and attack vectors
- Additional model integrations (Anthropic, Google, Cohere, etc.)
- Evaluation metrics and automated scoring
- Documentation improvements and tutorials
- Bug reports and fixes

## 🚀 Getting Started

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/jailbreak_harness.py.git`
3. **Create a branch**: `git checkout -b feature/amazing-feature`
4. **Make your changes**

## ✅ Development Setup
```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=src/jailbreak_harness

# Format code
black src/ tests/

# Lint
flake8 src/jailbreak_harness --max-line-length=100
```

## 📝 Code Standards

- **Follow PEP 8** style guidelines
- **Use `black`** for code formatting (max line length: 100)
- **Add type hints** to all function signatures
- **Write docstrings** for all public functions and classes
- **Include tests** for all new features
- **Update documentation** if you change functionality

## 🧪 Testing Requirements

All new features must include:
- Unit tests with >80% coverage
- Integration tests if applicable
- Test cases in the test suite YAML if adding attack patterns

## 📋 Pull Request Process

1. Ensure all tests pass: `pytest tests/ -v`
2. Update the README if you change functionality
3. Add your changes to the version changelog
4. Submit a PR with a clear description of changes
5. Link any related issues

## 🔒 Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md).
Do not open public issues for security problems.

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
