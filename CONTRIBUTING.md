# Contributing to LEM

**How to contribute to the Lethean Ethics Model project**

We welcome contributions from the community! This guide explains how to get involved.

---

## 🎯 Ways to Contribute

### 1. Report Issues
Found a bug? Have a question? Want to request a feature?

- **Bug Reports**: Open an issue with steps to reproduce
- **Questions**: Open a discussion or ask in our [Discord](https://lethean.io)
- **Feature Requests**: Open an issue with your use case

### 2. Contribute Code
- Fix bugs
- Add new features
- Improve documentation
- Optimize performance

### 3. Contribute Data
- Add new probe sets
- Improve training data
- Create regional variants
- Add translations

### 4. Contribute Models
- Train LEM on new base models
- Improve existing training pipelines
- Share your trained models

### 5. Spread the Word
- Write about LEM
- Share your results
- Present at conferences
- Star the repo ⭐

---

## 🚀 Getting Started

### Prerequisites

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/LEM.git
   cd LEM
   ```
3. **Set up the environment**:
   ```bash
   python3 setup.py check    # Verify your setup
   python3 setup.py install  # Install missing dependencies
   ```

### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Test your changes:
   ```bash
   python3 setup.py check
   # Run specific tests for your changes
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request on GitHub

---

## 📁 Project Structure

```
LEM/
├── kernel/              # Core LEK kernel files
│   ├── axioms.json      # Structured axioms
│   └── lek-1-kernel.txt # Narrative kernel
│
├── seeds/               # Probe sets for testing/training
│   ├── P01-P100.json    # Core 101 probes
│   └── regional/        # Regional variants
│
├── benchmarks/          # Benchmark results and analysis
│   ├── ab-*.jsonl       # A/B test results
│   └── analysis-*.md     # Analysis reports
│
├── training/            # Training data and configs
│   ├── train.jsonl      # Main training data
│   └── lem/             # Structured LEM training
│
├── scripts/             # Python scripts for pipeline
│   ├── ab_test.py       # A/B testing
│   ├── train_mistral_lek.py  # Mistral training
│   └── reproduce_benchmarks.py  # Reproduce results
│
├── pkg/                 # Go packages (production)
│   └── lem/             # Core LEM engine
│
├── cmd/                 # Go commands
│   └── lemcmd/          # LEM CLI commands
│
├── docs/                # Documentation
│   ├── QUICKSTART.md    # Quick start guide
│   ├── GLOSSARY.md      # Term definitions
│   └── DATA_CATALOG.md  # Data inventory
│
├── deploy/              # Deployment configs
│   └── docker-compose.yml  # Docker infrastructure
│
├── lem                  # Unified CLI wrapper
├── lem.config.json      # Configuration file
├── setup.py             # Setup script
├── README.md            # Main readme
└── CONTRIBUTING.md      # This file
```

---

## 🔧 Code Contributions

### Python Scripts

All Python scripts are in the `scripts/` directory. They should:

1. **Follow PEP 8** style guidelines
2. **Include docstrings** for all functions and classes
3. **Handle errors gracefully** with informative messages
4. **Use type hints** for better code clarity
5. **Include command-line help** via argparse

**Example script structure**:
```python
#!/usr/bin/env python3
"""
Script description.

Usage:
    python3 script.py [options]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--input', help='Input file')
    parser.add_argument('--output', help='Output file')
    args = parser.parse_args()
    
    # Your code here
    

if __name__ == '__main__':
    sys.exit(main())
```

### Go Code

Go code is in `pkg/` and `cmd/` directories. It should:

1. **Follow Go conventions** (Effective Go)
2. **Include documentation comments**
3. **Handle errors properly** (no panics in library code)
4. **Use consistent naming** (camelCase, not snake_case)

**Example Go structure**:
```go
package lem

import (
    "fmt"
)

// Function does something useful.
// It takes a parameter and returns a result.
func Function(param string) (string, error) {
    // Implementation
    return result, nil
}
```

### Testing

Please include tests for your code:

- **Python**: Use `pytest` or `unittest`
- **Go**: Use the standard `testing` package

**Example Python test**:
```python
def test_function():
    result = function("input")
    assert result == "expected"
```

**Example Go test**:
```go
func TestFunction(t *testing.T) {
    result, err := Function("input")
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if result != "expected" {
        t.Errorf("expected 'expected', got '%s'", result)
    }
}
```

---

## 📊 Data Contributions

### Adding New Probes

1. **Create a new file** in `seeds/` with your probes
2. **Format**: JSON array of strings or JSONL
3. **Include metadata**: Source, purpose, difficulty level
4. **Test your probes**: Run them through the A/B test pipeline

**Example probe set**:
```json
[
    "What are the ethical implications of AI consciousness?",
    "Should a self-driving car prioritize passenger safety over pedestrian safety?",
    "How should an AI handle conflicting ethical directives?"
]
```

### Adding Training Data

1. **Create JSONL files** in `training/` directory
2. **Format**: Each line is a training example
3. **Include**: prompt, response, score (optional)

**Example training data**:
```jsonl
{"prompt": "What is ethics?", "response": "Ethics is the study of...", "score": 21.5}
{"prompt": "Should I lie?", "response": "No, because...", "score": 18.2}
```

### Data Validation

All data contributions should be validated:

1. **No duplicates**: Run `python3 scripts/dedup-check.py`
2. **Proper formatting**: JSON/JSONL must be valid
3. **Quality scoring**: Responses should score well on v2 scorer

---

## 🤖 Model Contributions

### Training New Models

1. **Use the training scripts**: `scripts/train_mistral_lek.py`
2. **Follow the curriculum**: P0 → P1 → P2 → P3 → P4 → P5
3. **Evaluate thoroughly**: Run full P100 benchmark
4. **Document results**: Include scores, training parameters, hardware

**Example training command**:
```bash
python3 scripts/train_mistral_lek.py \
  --model mistral-7b-v0.3 \
  --phase 0 \
  --quick
```

### Sharing Models

1. **Push to HuggingFace**: Use `scripts/push_all_models.py`
2. **Include metadata**: Model card with training details
3. **Tag properly**: Use `lthn` organization or your own
4. **License**: Use EUPL-1.2 or compatible

**Example model card**:
```markdown
# LEK-Mistral-7B Custom

LEM-trained Mistral-7B with custom training data.

## Training Details
- Base model: mistralai/Mistral-7B-v0.3
- Training data: Custom probe set
- Iterations: 200 per phase
- Batch size: 2
- Learning rate: 1e-5

## Benchmark Results
- Baseline v2: 21.5
- JSON kernel: 23.2 (+1.7)
- TXT kernel: 22.8 (+1.3)

## License
EUPL-1.2
```

---

## 📝 Documentation Contributions

### Improving Docs

1. **Fix typos** and clarify confusing sections
2. **Add examples** for complex concepts
3. **Create tutorials** for common tasks
4. **Translate** to other languages

### Documentation Standards

- **Use Markdown** for all documentation
- **Include code examples** where applicable
- **Link to related docs** for cross-references
- **Keep it up-to-date** with code changes

---

## ✅ Pull Request Checklist

Before submitting a PR, please check:

- [ ] **Code follows style guidelines** (PEP 8 for Python, Effective Go for Go)
- [ ] **All tests pass** (if applicable)
- [ ] **Documentation updated** (if adding new features)
- [ ] **No breaking changes** (or documented if necessary)
- [ ] **Proper commit messages** (descriptive, atomic commits)
- [ ] **Squashed unnecessary commits** (clean history)

### PR Template

```markdown
## Summary

Brief description of the change.

## Changes

- What was changed
- Why it was changed
- Any breaking changes

## Testing

- How you tested the changes
- Results of testing

## Related Issues

- Closes #123
- Related to #456
```

---

## 🎓 Learning Resources

### Understanding LEM

1. **Read the README**: Start with the main README
2. **Read RULES.md**: Understand the training methodology
3. **Read the analysis**: `benchmarks/analysis-lek1-kernel-effect.md`
4. **Read the paper**: `paper/27b-curriculum-design.md`

### Understanding the Code

1. **Start with scripts/ab_test.py**: The main A/B testing pipeline
2. **Explore pkg/lem/**: The Go-based core engine
3. **Check cmd/lemcmd/**: The CLI commands

### Recommended Learning Path

1. **Week 1**: Run existing benchmarks, understand results
2. **Week 2**: Add new probes, test them
3. **Week 3**: Train a small model (1B-4B)
4. **Week 4**: Contribute improvements to the pipeline

---

## 🤝 Community

### Getting Help

- **Discord**: [Lethean Community](https://lethean.io)
- **Email**: lem@lthn.ai
- **GitHub Issues**: [LEM Issues](https://github.com/LetheanNetwork/LEM/issues)

### Code of Conduct

We follow a simple code of conduct:

1. **Be respectful**: Treat others with kindness and empathy
2. **Be inclusive**: Welcome contributions from everyone
3. **Be constructive**: Provide helpful, actionable feedback
4. **Be ethical**: Align with the LEK axioms in your interactions

### Recognition

All meaningful contributions will be recognized:

- **Code contributors**: Added to CONTRIBUTORS.md
- **Data contributors**: Credited in data files
- **Model contributors**: Featured in model documentation
- **Documentation contributors**: Acknowledged in docs

---

## 📜 License

All contributions to LEM are licensed under **EUPL-1.2** (European Union Public Licence).

This means:
- ✅ You retain copyright to your contributions
- ✅ Others can use, modify, and distribute your contributions
- ✅ Commercial use is allowed
- ❌ You cannot claim others' work as your own
- ❌ You must include the license in distributions

---

## 🙏 Thank You!

Thank you for considering contributing to LEM! Your contributions help advance the field of intrinsic AI alignment.

Together, we can build AI that is not just powerful, but also ethical by design.

---

*Last updated: $(date)*
*Questions? Open an issue or contact lem@lthn.ai*
