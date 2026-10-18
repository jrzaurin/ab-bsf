#!/usr/bin/env bash
set -euo pipefail

# `--check` (CI) reports what isort/black would change instead of changing it
CHECK=""
[[ "${1:-}" == "--check" ]] && CHECK=1

# sort imports
isort --quiet ${CHECK:+--check-only --diff} .

# black code style
black ${CHECK:+--check --diff} .

# flake8 standards
flake8 . --max-complexity=10 --max-line-length=127 --ignore=E203,E266,E501,E722,E721,F401,F403,F405,W503,C901,F811 --exclude=.venv

# mypy
mypy ab_bsf --ignore-missing-imports --no-strict-optional
