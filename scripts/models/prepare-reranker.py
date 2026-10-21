#!/usr/bin/env python3
"""Fetch pinned upstream reranker inputs for offline conversion, outside the model cache."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open('rb') as stream:
        while data := stream.read(1024 * 1024):
            sha.update(data)
            blob.update(data)
    return (sha.hexdigest() == entry['sha256'] if entry['sha256']
            else blob.hexdigest() == entry['git_blob_sha1'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('reranker-source.json').read_text())
    for entry in manifest['files']:
        relative = Path(entry['name'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid source path')
        path = args.destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if verify(path, entry):
            print(f"Verified existing {relative}", flush=True)
            continue
        if path.exists():
            raise ValueError(f'refusing to replace invalid existing source: {path}')
        partial = path.with_name(path.name + '.part')
        url = (f"https://huggingface.co/{manifest['model']}/resolve/"
               f"{manifest['revision']}/{relative.as_posix()}")
        request = urllib.request.Request(url, headers={'User-Agent': 'dkc/0.1.0'})
        print(f"Downloading {relative}: {entry['size']} bytes", flush=True)
        count = 0
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
            while data := response.read(1024 * 1024):
                count += len(data)
                if count > entry['size']:
                    raise ValueError(f'oversize download: {relative}')
                stream.write(data)
        if not verify(partial, entry):
            raise ValueError(f'source verification failed: {relative}')
        partial.rename(path)
        print(f"Verified {relative}", flush=True)
    print(f"All inputs verified at {manifest['revision']}", flush=True)


if __name__ == '__main__':
    main()
