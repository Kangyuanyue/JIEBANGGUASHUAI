#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash miao/scripts/push_to_github.sh
# Run this script from the root of https://github.com/Kangyuanyue/JIEBANGGUASHUAI.git

if [ ! -d ".git" ]; then
  echo "ERROR: please run this script from the repository root." >&2
  exit 1
fi

if [ ! -d "miao" ]; then
  echo "ERROR: miao/ folder not found." >&2
  exit 1
fi

git status
git add miao
git commit -m "Add miao solution for anti-interference speech command recognition" || {
  echo "Nothing to commit or commit failed. Check git status." >&2
  exit 1
}
git push origin main
