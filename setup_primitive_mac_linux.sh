#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v go >/dev/null || { echo 'Install Go from https://go.dev/dl/ first'; exit 1; }
go install github.com/fogleman/primitive@latest
mkdir -p tools
cp "$(go env GOPATH)/bin/primitive" tools/primitive
chmod +x tools/primitive
echo 'Installed Primitive into tools/primitive'
