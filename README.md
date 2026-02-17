# Project Iron

## Directory Structure

```
project-iron/
├── Dockerfile                  ← Docker setup (3 layers)
├── locking-requirements.txt    ← Bare metal pip fallback
├── conda_environment.yaml      ← Bare metal conda fallback
├── .pre-commit-config.yaml     ← black + flake8 hooks
├── .gitignore
├── setup.py
│
├── src/                        ← All Python source code
│   └── main.py
├── models/                     ← Saved .pt / .onnx / OpenVINO files
├── data/                       ← Raw & processed datasets
├── notebooks/                  ← Jupyter experiments
├── tests/                      ← pytest test files
├── configs/                    ← YAML/JSON config files
├── scripts/                    ← Shell/utility scripts
└── docs/                       ← Documentation
```

---

## Option A — Docker (Recommended)

```bash
# Build
docker build -t project-iron .

# Run
docker run --rm -it project-iron
```

---

## Option B — Bare Metal (Low RAM fallback)

### With Conda (preferred)
```bash
conda env create -f conda_environment.yaml
conda activate project-iron
```

### With pip only
```bash
pip install -r locking-requirements.txt
```

---

## Git Setup + Pre-commit Hooks

```bash
git init
git add .
git commit -m "chore: initial Project Iron scaffold"

# Install hooks (runs black + flake8 on every commit)
pip install pre-commit
pre-commit install

# Test hooks manually
pre-commit run --all-files
```

---

## Why Two Environments?

| | Docker | Conda/pip |
|---|---|---|
| Isolation | Perfect | Good |
| RAM overhead | ~200–400MB | Minimal |
| Best for | CI/CD, deployment | D4RT dev on low-RAM machines |
