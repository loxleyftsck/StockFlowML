# Contributing to StockFlowML

Thank you for your interest in contributing to StockFlowML! This document provides guidelines and instructions for contributing to the project.

## 🤝 Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/StockFlowML.git`
3. **Create** a feature branch from `development`: `git checkout -b feature/my-feature`
4. **Make** your changes and commit them
5. **Push** to your fork and open a **Pull Request** to `development`

## 📋 Table of Contents

- [Development Setup](#development-setup)
- [Branching Strategy](#branching-strategy)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Messages](#commit-messages)

## 🛠️ Development Setup

### Prerequisites
- Python 3.11+
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/StockFlowML.git
cd StockFlowML

# Set up upstream remote
git remote add upstream https://github.com/loxleyftsck/StockFlowML.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Verify installation
pytest tests/
```

## 🌳 Branching Strategy

We follow a **Git Flow** style branching strategy. See [BRANCHING_STRATEGY.md](docs/BRANCHING_STRATEGY.md) for detailed information.

### Branch Types

| Branch | Purpose | Base | Merge To |
|--------|---------|------|----------|
| `main` | Production code | - | - |
| `development` | Integration | - | `main` |
| `feature/*` | New features | `development` | `development` |
| `bugfix/*` | Bug fixes | `development` | `development` |
| `hotfix/*` | Production fixes | `main` | `main` + `development` |
| `experiment/*` | Experiments | `development` | Optional |

### Creating a Feature Branch

```bash
# Start from latest development
git checkout development
git pull upstream development

# Create your feature branch
git checkout -b feature/my-awesome-feature

# Make changes, commit, push
git add .
git commit -m "feat: add awesome feature"
git push origin feature/my-awesome-feature
```

## 💻 Coding Standards

### Python Style Guide

We follow **PEP 8** with the following tools:

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy (optional but recommended)
mypy src/
```

### Code Quality Checklist

- [ ] Code follows PEP 8
- [ ] All functions have docstrings
- [ ] Type hints are used where appropriate
- [ ] No commented-out code
- [ ] No print statements (use logging)
- [ ] Error handling is comprehensive
- [ ] Code is DRY (Don't Repeat Yourself)

### Documentation

- Update README.md if adding new features
- Add docstrings to all public functions/classes
- Update relevant documentation in `docs/`
- Add usage examples where applicable

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_validation.py

# Run with verbose output
pytest tests/ -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names
- Follow AAA pattern: Arrange, Act, Assert

Example:
```python
def test_data_validation_catches_missing_columns():
    """Test that validation detects missing required columns."""
    # Arrange
    df = pd.DataFrame({'close': [100, 101, 102]})
    
    # Act
    is_valid, errors = validate_data(df)
    
    # Assert
    assert not is_valid
    assert 'Missing required columns' in errors[0]
```

### Test Coverage Requirements

- Minimum coverage: **80%**
- Critical modules (data, models, monitoring): **90%+**

## 🔄 Pull Request Process

### Before Opening a PR

1. **Sync with upstream**:
   ```bash
   git checkout development
   git pull upstream development
   git checkout feature/my-feature
   git rebase development
   ```

2. **Run tests**:
   ```bash
   pytest tests/
   ```

3. **Check code quality**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

4. **Update documentation** if needed

### Opening the PR

1. **Push to your fork**:
   ```bash
   git push origin feature/my-feature
   ```

2. **Open PR** on GitHub from `feature/my-feature` to `development`

3. **Fill out PR template** completely:
   - Description of changes
   - Type of change (feature/bugfix/docs)
   - Testing performed
   - Checklist items

4. **Link related issues** using keywords: `Fixes #123`, `Relates to #456`

### PR Title Format

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat: add drift detection module
fix: resolve encoding issue on Windows
docs: update README with Level 2 features
test: add integration tests for pipeline
```

### PR Review Process

1. **Automated CI checks** run (must pass)
2. **Code review** by maintainer(s)
3. **Address feedback** and update PR
4. **Approval** from at least 1 maintainer
5. **Merge** to development

## 📝 Commit Messages

We use **Conventional Commits** specification.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance
- `perf`: Performance
- `ci`: CI/CD

### Examples

```bash
# Simple feature
git commit -m "feat: add Discord alerts for drift detection"

# With scope
git commit -m "fix(data): handle missing values in OHLCV data"

# With body
git commit -m "feat(monitoring): implement Evidently AI integration

- Add DriftDetector class
- Generate HTML/JSON reports
- Integrate with alert system

Closes #42"

# Breaking change
git commit -m "feat(api)!: change response format

BREAKING CHANGE: API now returns camelCase instead of snake_case"
```

## 🎯 Code Review Guidelines

### For Contributors

- Respond to feedback promptly
- Be open to suggestions
- Ask questions if unclear
- Update PR based on feedback
- Mark conversations as resolved when addressed

### For Reviewers

- Be constructive and respectful
- Explain reasoning behind suggestions
- Approve if minor changes needed
- Request changes if significant issues
- Test the changes locally if needed

## 🐛 Bug Reports

### Before Reporting

1. Check if bug already reported in [Issues](https://github.com/loxleyftsck/StockFlowML/issues)
2. Verify bug exists on latest `development` branch
3. Gather reproduction steps

### Bug Report Template

```markdown
**Describe the bug**
Clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. With data '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- StockFlowML version/commit: [e.g., v1.0.0 or commit hash]

**Additional context**
Logs, screenshots, or any other context.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
Description of the problem or limitation.

**Describe the solution you'd like**
Clear description of what you want to happen.

**Describe alternatives you've considered**
Other approaches you've thought about.

**Additional context**
Mockups, examples, or related features in other projects.
```

## 📜 Project Structure

Understanding the project structure helps you know where to make changes:

```
StockFlowML/
├── src/                    # Source code
│   ├── data/              # Data ingestion & validation
│   ├── features/          # Feature engineering
│   ├── models/            # Model training
│   ├── evaluation/        # Metrics & evaluation
│   └── monitoring/        # Drift detection & alerts
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── pipelines/             # End-to-end pipelines
├── docs/                  # Documentation
└── reports/               # Generated reports
```

## 🏆 Recognition

Contributors are recognized in:
- README.md acknowledgments
- GitHub contributors page
- Release notes

## 📞 Getting Help

- **Questions**: Open a [Discussion](https://github.com/loxleyftsck/StockFlowML/discussions)
- **Bugs**: Open an [Issue](https://github.com/loxleyftsck/StockFlowML/issues)
- **Chat**: Join our [Discord](#) (if applicable)

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

---

**Thank you for contributing to StockFlowML!** 🚀

Every contribution, no matter how small, helps make this project better for everyone.
