#!/usr/bin/env bash
set -euo pipefail

git add .
git commit -m "update"
git push

echo "=== Changes pushed and GitHub Actions triggered ==="
