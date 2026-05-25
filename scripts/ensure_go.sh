#!/usr/bin/env bash
# Ensure Go >= 1.22 on Linux (Colab apt ships go1.18, which cannot parse go 1.22.1 in go.mod).
# Source or call from install_colab_deps.sh / render_projections.sh.
set -euo pipefail

GO_VERSION="${GO_VERSION:-1.22.10}"
GO_ROOT="${GO_ROOT:-/usr/local/go}"

_go_ok() {
  command -v go >/dev/null 2>&1 || return 1
  go version 2>/dev/null | grep -qE 'go1\.(2[2-9]|[3-9][0-9])'
}

if _go_ok; then
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "warning: Go >= 1.22 required; install manually (non-Linux host)" >&2
  exit 0
fi

echo "Installing Go ${GO_VERSION} to ${GO_ROOT}..."
curl -fL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go${GO_VERSION}.tar.gz
rm -rf "${GO_ROOT}"
tar -C /usr/local -xzf /tmp/go${GO_VERSION}.tar.gz
export PATH="${GO_ROOT}/bin:${PATH}"
go version
