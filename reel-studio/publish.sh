#!/usr/bin/env bash
# Publish this standalone bundle to a GitHub repository.
#
# Run from inside the `reel-studio/` directory, on a machine where you are
# authenticated to GitHub (gh CLI logged in, or a git credential helper set up).
#
#   cd reel-studio
#   ./publish.sh [repo-name] [github-username]
#
# Works whether the repo is brand-new/empty OR was created with a README.
# Defaults: repo-name=reel-studio, username=mhassanien2
set -euo pipefail

REPO_NAME="${1:-reel-studio}"
GH_USER="${2:-mhassanien2}"
REMOTE_URL="https://github.com/$GH_USER/$REPO_NAME.git"

git init -q
git add -A
git commit -q -m "Reel Studio — Instagram Reel production package generator" || true
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

echo "==> Pushing to $REMOTE_URL"
if ! git push -u origin main 2>/dev/null; then
  echo "==> Remote already has commits (e.g. an auto-created README); rebasing onto it"
  git pull --rebase origin main
  git push -u origin main
fi
echo "==> Done: https://github.com/$GH_USER/$REPO_NAME"
