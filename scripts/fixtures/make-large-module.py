#!/usr/bin/env python3
"""Generate a module collation above 256 KiB without committing a large fixture."""
from pathlib import Path
import sys

root = Path(sys.argv[1])
assert root.is_dir()
long_name = "entry" * 400
for number in range(60):
    path = root / f"large_{number:03}.rs"
    with path.open("x") as output:
        output.write("".join(f"pub fn {long_name}_{item}() {{}}\n" for item in range(3)))
