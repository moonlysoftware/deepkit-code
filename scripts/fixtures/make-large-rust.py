#!/usr/bin/env python3
"""Emit an exactly sized Rust source fixture without storing large files in Git."""
import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bytes", type=int)
args = parser.parse_args()
prefix = b"pub fn checkoutmarker() { let large_payload = 3; }\n"
if args.bytes < len(prefix):
    parser.error(f"size must be at least {len(prefix)} bytes")
sys.stdout.buffer.write(prefix)
remaining = args.bytes - len(prefix)
while remaining:
    count = min(remaining, 65536)
    sys.stdout.buffer.write(b" " * count)
    remaining -= count
