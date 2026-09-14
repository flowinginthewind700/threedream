#!/usr/bin/env bash
# Fetch the Unreal Engine 5 source tree into thirdparty/UnrealEngine as a git
# submodule, for reading as an architecture reference.
#
# REQUIREMENTS: your GitHub account must already have accepted Epic's EULA and
# been granted access to the private EpicGames/UnrealEngine repository. There is
# no public mirror, and this script does not create one.
#
# Usage: bash scripts/clone_unreal_reference.sh [branch]   (default: release)
set -uo pipefail

BRANCH="${1:-release}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$REPO_ROOT/thirdparty/UnrealEngine"

# Epic's repo is ~39 GB of full history. Depth 1 is plenty for reading source,
# and the shallow flag is recorded in .gitmodules so future inits stay shallow.
GIT_OPTS=(
  -c http.postBuffer=524288000
  -c http.lowSpeedLimit=0
  -c http.lowSpeedTime=999999
  -c http.version=HTTP/1.1
  -c core.compression=9
)

cd "$REPO_ROOT" || exit 1

echo "==> Unreal Engine reference submodule (branch: $BRANCH)"
echo "    Target: $TARGET"
echo "    This is a multi-GB download. Retrying on transient network errors."

if [ -d "$TARGET/.git" ] || [ -f "$TARGET/.git" ]; then
  echo "==> Already present. Updating."
  git "${GIT_OPTS[@]}" submodule update --init --depth 1 thirdparty/UnrealEngine && exit 0
fi

if git "${GIT_OPTS[@]}" config --file .gitmodules --get submodule.thirdparty/UnrealEngine.url >/dev/null 2>&1; then
  echo "==> .gitmodules entry exists; running submodule update."
  git "${GIT_OPTS[@]}" submodule update --init --depth 1 thirdparty/UnrealEngine && exit 0
fi

attempt=1
max_attempts="${MAX_ATTEMPTS:-6}"
while [ "$attempt" -le "$max_attempts" ]; do
  echo "==> Attempt $attempt/$max_attempts"
  if git "${GIT_OPTS[@]}" submodule add --force --depth 1 --branch "$BRANCH" \
      https://github.com/EpicGames/UnrealEngine.git thirdparty/UnrealEngine; then
    # Record shallow so `git submodule update --init` by others stays depth 1.
    git config --file .gitmodules submodule.thirdparty/UnrealEngine.shallow true
    echo "==> Success."
    exit 0
  fi
  echo "==> Attempt $attempt failed."
  attempt=$((attempt + 1))
  sleep 15
done

echo "!! Could not clone Unreal Engine after $max_attempts attempts." >&2
echo "   Check that your GitHub account has EULA access:" >&2
echo "   gh api repos/EpicGames/UnrealEngine" >&2
exit 1
