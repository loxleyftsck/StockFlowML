# Git Branching Strategy & Workflow

> **Professional branching strategy for StockFlowML MLOps project**

## 🌳 Branch Overview

### Main Branches

#### `main` - Production Branch
- **Purpose**: Production-ready, stable code
- **Protection**: Protected branch with required reviews
- **Deployment**: Auto-deploys to production (when applicable)
- **Merge Strategy**: Only merge from `development` after validation
- **CI/CD**: Full test suite + deployment pipeline

#### `development` - Integration Branch
- **Purpose**: Integration and testing of new features
- **Protection**: Protected with CI checks
- **Testing**: All features must pass CI before merge to `main`
- **Merge Strategy**: Feature branches merge here first
- **CI/CD**: Full test suite + additional validation

### Supporting Branches

#### `feature/*` - Feature Development
- **Naming**: `feature/drift-detection`, `feature/api-endpoints`
- **Base Branch**: `development`
- **Merge Target**: `development`
- **Lifetime**: Temporary (deleted after merge)
- **Example**: `feature/level3-serving`

#### `bugfix/*` - Bug Fixes
- **Naming**: `bugfix/fix-data-loader`, `bugfix/encoding-error`
- **Base Branch**: `development`
- **Merge Target**: `development`
- **Lifetime**: Temporary (deleted after merge)
- **Example**: `bugfix/windows-compatibility`

#### `hotfix/*` - Production Hotfixes
- **Naming**: `hotfix/critical-security-patch`
- **Base Branch**: `main`
- **Merge Target**: `main` AND `development`
- **Lifetime**: Temporary (deleted after merge)
- **Use Case**: Critical production issues only

#### `experiment/*` - Experimental Work
- **Naming**: `experiment/xgboost-tuning`, `experiment/new-features`
- **Base Branch**: `development`
- **Merge Target**: Optional (may be discarded)
- **Lifetime**: Temporary or long-lived
- **Use Case**: Exploratory work, A/B testing

---

## 🔄 Workflow Diagrams

### Standard Feature Development Flow
```
development → feature/my-feature → [develop] → [test] → development → [validate] → main
```

### Hotfix Flow
```
main → hotfix/critical-fix → main
                           → development (backport)
```

---

## 📋 Development Workflow

### 1. Starting New Feature

```bash
# Ensure you're on latest development
git checkout development
git pull origin development

# Create feature branch
git checkout -b feature/my-new-feature

# Work on your feature
# ... make changes ...

# Commit with conventional commits
git add .
git commit -m "feat: add new feature description"

# Push to remote
git push -u origin feature/my-new-feature
```

### 2. Opening Pull Request

**From**: `feature/my-new-feature`  
**To**: `development`

**PR Template** (see `.github/PULL_REQUEST_TEMPLATE.md`):
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Feature
- [ ] Bug fix
- [ ] Documentation
- [ ] Refactor

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### 3. Merging to Development

**CI Quality Gate** must pass:
- ✅ All unit tests pass
- ✅ Integration tests pass
- ✅ Linting checks pass
- ✅ Code coverage above threshold
- ✅ No critical security vulnerabilities

**Review Requirements**:
- At least 1 approval (for solo projects, self-review with checklist)
- No unresolved discussions
- CI passing

```bash
# After PR approval
git checkout development
git pull origin development
git merge feature/my-new-feature --no-ff
git push origin development

# Delete feature branch
git branch -d feature/my-new-feature
git push origin --delete feature/my-new-feature
```

### 4. Promoting to Production (main)

**Validation Steps**:
1. All features in `development` tested
2. Performance metrics meet baselines
3. No drift detected in model validation
4. Documentation updated

**Merge to Main**:
```bash
# Create release PR
git checkout main
git pull origin main

# Merge development into main
git merge development --no-ff -m "chore: release v1.x.x"

# Tag the release
git tag -a v1.x.x -m "Release version 1.x.x"
git push origin main --tags
```

---

## 🛡️ Branch Protection Rules

### `main` Branch Protection

**Required**:
- ✅ Require pull request before merging
- ✅ Require status checks to pass
  - `test` - All tests must pass
  - `lint` - Code quality checks
  - `drift-check` - Model drift validation (if applicable)
- ✅ Require conversation resolution
- ✅ Require linear history (no merge commits from PRs)
- ✅ Block force pushes
- ✅ Block deletions

**Recommended**:
- Require code owner review (when team grows)
- Require signed commits

### `development` Branch Protection

**Required**:
- ✅ Require status checks to pass
  - `test` - All tests must pass
  - `lint` - Code quality checks
- ✅ Block force pushes
- ✅ Block deletions

**Optional**:
- Require pull request (recommended for teams)

---

## 📝 Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvement
- `ci`: CI/CD changes

### Examples
```bash
# Feature
git commit -m "feat(monitoring): add drift detection with Evidently AI"

# Bug fix
git commit -m "fix(data): handle missing values in feature engineering"

# Documentation
git commit -m "docs: update README with Level 2 features"

# Breaking change
git commit -m "feat(api)!: change response format

BREAKING CHANGE: API responses now use camelCase"
```

---

## 🔍 CI/CD Integration

### GitHub Actions Workflow

**On Push to `development`**:
```yaml
- Run tests
- Run linting
- Generate coverage report
- Build artifacts
- Run drift detection (if model changes)
```

**On PR to `main`**:
```yaml
- All development checks PLUS:
- Performance validation
- Security scan
- Documentation check
- Release notes generation
```

**On Tag `v*`**:
```yaml
- Create GitHub Release
- Build production artifacts
- Deploy to production (if applicable)
```

---

## 🎯 Best Practices

### DO ✅
- Always branch from `development` for features
- Keep feature branches short-lived (< 1 week)
- Squash commits before merging if history is messy
- Write descriptive commit messages
- Update documentation with code changes
- Run tests locally before pushing
- Review your own PR before requesting review

### DON'T ❌
- Don't commit directly to `main`
- Don't push broken code to `development`
- Don't merge without CI passing
- Don't force push to protected branches
- Don't leave stale branches (clean up after merge)
- Don't mix multiple features in one PR

---

## 🚨 Emergency Procedures

### Critical Production Bug

```bash
# 1. Create hotfix from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-bug-fix

# 2. Fix the issue
# ... make changes ...

# 3. Test thoroughly
pytest tests/

# 4. Commit and push
git commit -m "fix: critical bug in production"
git push -u origin hotfix/critical-bug-fix

# 5. Open PR to main (bypass normal flow if critical)
# 6. After merge to main, backport to development
git checkout development
git pull origin development
git merge hotfix/critical-bug-fix
git push origin development

# 7. Delete hotfix branch
git branch -d hotfix/critical-bug-fix
git push origin --delete hotfix/critical-bug-fix
```

---

## 📊 Branch Lifecycle

```
feature/my-feature
├── Created from: development
├── Lifetime: 1-7 days
├── Commits: 1-20
├── Merge to: development
└── Then: Delete

development
├── Always exists
├── Receives: feature/* bugfix/* experiment/*
├── Promotes to: main
└── Protected: Yes (CI checks)

main
├── Always exists
├── Receives: development (via PR)
├── Tagged: v1.0.0, v1.1.0, etc.
└── Protected: Yes (strict)
```

---

## 🔧 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/loxleyftsck/StockFlowML.git
cd StockFlowML
```

### 2. Configure Git
```bash
# Set up user info (if not already done)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Recommended: Enable autostash
git config pull.rebase true
git config rebase.autoStash true
```

### 3. Verify Branches
```bash
# List all branches
git branch -a

# Should see:
# * development
#   main
#   remotes/origin/development
#   remotes/origin/main
```

### 4. Start Working
```bash
# Always start from development
git checkout development
git pull origin development

# Create your feature branch
git checkout -b feature/your-feature-name
```

---

## 📚 Additional Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Semantic Versioning](https://semver.org/)
- [Git Best Practices](https://git-scm.com/book/en/v2)

---

## 📞 Support

For questions about branching strategy:
1. Check this document first
2. Review existing PRs for examples
3. Open a discussion in GitHub Discussions
4. Contact project maintainers

---

**Last Updated**: 2026-02-01  
**Version**: 1.0.0  
**Status**: Active
