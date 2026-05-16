#!/usr/bin/env bash
# Download + decompress the Push-T expert dataset into $STABLEWM_HOME.
#
# Source: https://huggingface.co/datasets/quentinll/lewm-pusht
# File:   pusht_expert_train.h5.zst  (~13 GB compressed; ~25-35 GB decompressed)
#
# Disk: budget ~50 GB free in $STABLEWM_HOME during decompression.
#
# Usage:
#   STABLEWM_HOME=/path/to/storage bash scripts/download_pusht.sh

set -euo pipefail

STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
mkdir -p "$STABLEWM_HOME"
cd "$STABLEWM_HOME"

URL="https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst"
ARCHIVE="pusht_expert_train.h5.zst"
TARGET="pusht_expert_train.h5"

if [[ -f "$TARGET" ]]; then
  echo "Already have $STABLEWM_HOME/$TARGET — nothing to do."
  exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
  echo "Downloading $URL ..."
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -o "$ARCHIVE" "$URL"
  else
    curl -L --fail -o "$ARCHIVE" "$URL"
  fi
fi

echo "Decompressing $ARCHIVE -> $TARGET ..."
if command -v zstd >/dev/null 2>&1; then
  zstd -d --rm "$ARCHIVE" -o "$TARGET"
else
  echo "zstd not installed. brew install zstd  (or: conda install -c conda-forge zstd)" >&2
  exit 1
fi

echo "Done: $STABLEWM_HOME/$TARGET"
