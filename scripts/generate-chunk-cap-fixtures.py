#!/usr/bin/env python3
"""Generate hostile chunk-cap sources outside the repository (v1 sections 3.5/6)."""
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
count = 3000
(root / 'many.rs').write_text(''.join(f'pub fn item{i}() {{ work(); }}\n' for i in range(count)))
(root / 'long.rs').write_text('pub fn giant() { ' + 'x + ' * count + 'x; }\n')
(root / 'many.md').write_text(''.join(f'# Section {i}\n\nSome text about behaviour.\n\n' for i in range(count)))
(root / 'many.txt').write_text('A paragraph about behaviour and testing.\n\n' * count)
(root / 'many.json').write_text(json.dumps({f'key{i:04}': i for i in range(count)}))
(root / 'many.toml').write_text(''.join(f'key{i:04} = {i}\n' for i in range(count)))
(root / 'many.yaml').write_text(''.join(f'key{i:04}: {i}\n' for i in range(count)))
(root / '.env').write_text(''.join(f'KEY{i:04}={i}\n' for i in range(count)))
