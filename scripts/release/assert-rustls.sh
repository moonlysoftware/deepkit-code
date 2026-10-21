#!/usr/bin/env bash
# Release dependency guard for v1.md §12: HTTP TLS must use rustls only.
set -euo pipefail

tree="$(cargo tree --locked --edges normal)"
if grep -Eq '(^|[[:space:]├└│])((openssl|openssl-sys|native-tls|hyper-tls) v)' <<<"$tree"; then
  echo "release dependency graph contains an OpenSSL/native-tls implementation" >&2
  exit 1
fi
if ! grep -Eq '(^|[[:space:]├└│])rustls v' <<<"$tree"; then
  echo "release dependency graph does not contain rustls" >&2
  exit 1
fi
echo "release dependency graph uses rustls without OpenSSL/native-tls"
