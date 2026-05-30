#!/usr/bin/env bash
# Publish this standalone bundle as its own GitHub repository.
#
# Run this from inside the `reel-studio/` directory, on a machine where you are
# authenticated to GitHub (gh CLI logged in, or a git credential helper set up).
#
#   cd reel-studio
#   ./publish.sh [repo-name] [github-username]
#
# Defaults: repo-name=reel-studio, username=mhassanien2
set -euo pipefail

REPO_NAME="${1:-reel-studio}"
GH_USER="${2:-mhassanien2}"

echo "==> Initializing git in $(pwd)"
git init -q
git add -A
git commit -q -m "Initial commit: Reel Studio — Instagram Reel production package generator"
git branch -M main

if command -v gh >/dev/null 2>&1; then
  echo "==> Creating GitHub repo $GH_USER/$REPO_NAME via gh and pushing"
  gh repo create "$GH_USER/$REPO_NAME" --public --source=. --remote=origin --push
  echo "==> Done: https://github.com/$GH_USER/$REPO_NAME"
else
  echo "gh CLI not found. Create the empty repo on github.com first, then run:"
  echo "    git remote add origin https://github.com/$GH_USER/$REPO_NAME.git"
  echo "    git push -u origin main"
fi
