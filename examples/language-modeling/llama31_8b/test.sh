#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    input="$1"
else
    input="/dev/stdin"
fi

grep ':::MLLOG' "$input" \
| sed 's/^.*:::MLLOG //' \
| jq -r '"\(.key)=\(.value)"'
