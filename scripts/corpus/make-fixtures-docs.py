#!/usr/bin/env python3
"""Build and verify the pinned document corpus; plan/05-corpus.md §4, v1 §6/§16."""
import argparse
import hashlib
import html
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import time
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
EPOCH = 'D:20200101000000Z'
GIT_SEED_MESSAGE = 'Seed generated document fixture corpus'
EMPTY_TREE = '4b825dc642cb6eb9a060e54bf8d69288fbee4904'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def relative(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'unsafe manifest path: {name}')
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'manifest path escapes root: {name}')
    return target


def command(args, **kwargs):
    result = subprocess.run([str(arg) for arg in args], capture_output=True, **kwargs)
    if result.returncode:
        detail = result.stderr if isinstance(result.stderr, str) else result.stderr.decode(errors='replace')
        print(detail[-8000:])
        result.check_returncode()
    return result


def verify_git(root):
    """Require a seeded Git HEAD so the generated target can enter bench run."""
    if not (root / '.git').is_dir() or (root / '.git').is_symlink():
        raise ValueError('generated fixture has no local Git repository')
    def read(*args):
        return command(['git', '-C', root, *args], text=True).stdout.strip()
    head = read('rev-parse', 'HEAD')
    if (read('rev-parse', 'HEAD^{tree}') != EMPTY_TREE or
            read('show', '-s', '--format=%s', 'HEAD') != GIT_SEED_MESSAGE or
            read('show', '-s', '--format=%P', 'HEAD') or
            read('ls-files', '--cached')):
        raise ValueError('generated fixture Git HEAD is not the empty seed commit')
    return head


def seed_git(root):
    """Create a local empty commit without tracking huge or hostile fixtures."""
    git_dir = root / '.git'
    if git_dir.is_symlink() or (git_dir.exists() and not git_dir.is_dir()):
        raise ValueError('generated fixture Git directory is not local')
    if not git_dir.exists():
        command(['git', 'init', '--quiet', '--initial-branch=main', root])
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', '--verify', 'HEAD'],
                          capture_output=True)
    if head.returncode == 0:
        return verify_git(root)
    if command(['git', '-C', root, 'ls-files', '--cached'], text=True).stdout:
        raise ValueError('generated fixture Git index is not empty')
    env = os.environ.copy()
    env.update({'GIT_AUTHOR_DATE': '2020-01-01T00:00:00+00:00',
                'GIT_COMMITTER_DATE': '2020-01-01T00:00:00+00:00',
                'GIT_AUTHOR_NAME': 'Deepkit Code Fixtures',
                'GIT_COMMITTER_NAME': 'Deepkit Code Fixtures',
                'GIT_AUTHOR_EMAIL': 'fixtures@deepkit.invalid',
                'GIT_COMMITTER_EMAIL': 'fixtures@deepkit.invalid'})
    command(['git', '-C', root, '-c', 'user.name=Deepkit Code Fixtures',
             '-c', 'user.email=fixtures@deepkit.invalid',
             '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '--quiet',
             '-m', GIT_SEED_MESSAGE], env=env)
    return verify_git(root)


def put(target, data):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError(f'existing fixture differs: {target}')
        return
    part = target.with_name(target.name + '.part')
    part.write_bytes(data)
    part.replace(target)


def excerpt(root, row, rev):
    source = relative(root, row['source'])
    if digest(source) != row['sha256']:
        raise ValueError(f'Markdown source changed: {source}')
    text = source.read_text()
    title = re.search(r'^title:\s*(.+)$', text, re.M)
    title = title.group(1).strip('"\'') if title else source.parent.name
    if text.startswith('---\n'):
        text = text.split('---', 2)[-1]
    # Short source excerpts keep each generated manual section on one page.
    text = re.sub(r'!\[[^]]*\]\([^)]*\)', '', text)
    text = ' '.join(text.split())[:800]
    return {'title': title, 'text': text,
            'url': f'https://github.com/mdn/content/blob/{rev}/{row["source"]}'}


def normalized_pdf(source, target, expected_pages):
    import pikepdf
    with pikepdf.open(source) as pdf:
        if len(pdf.pages) != expected_pages:
            raise ValueError(f'{target.name}: {len(pdf.pages)} pages, expected {expected_pages}')
        pdf.docinfo['/CreationDate'] = EPOCH
        pdf.docinfo['/ModDate'] = EPOCH
        pdf.docinfo['/Producer'] = 'dkc fixture corpus'
        if '/Metadata' in pdf.Root:
            del pdf.Root['/Metadata']
        output = BytesIO()
        pdf.save(output, deterministic_id=True)
    put(target, output.getvalue())


def render_html(text, target, pages, scratch):
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        work = Path(directory)
        source = work / 'input.html'; source.write_text(text)
        pdf = work / 'output.pdf'
        command(['chromium', '--headless', '--disable-gpu', '--disable-dev-shm-usage',
                 '--no-pdf-header-footer', '--host-resolver-rules=MAP * ~NOTFOUND',
                 f'--user-data-dir={work / "profile"}', f'--print-to-pdf={pdf}', source.as_uri()])
        normalized_pdf(pdf, target, pages)


def verify(root, manifest, source_key):
    record = json.loads((root / '.deepkit/fixture-manifest.json').read_text())
    if record['source_key'] != source_key:
        raise ValueError('fixture inputs changed')
    def key(row):
        return row['target'] if 'target' in row else 'hex:' + row['target_bytes_hex']
    expected = {'00-digital/' + row['id'] + '.pdf': row['sha256'] for row in manifest['paper']}
    expected.update({row['target']: row['sha256'] for row in manifest['image']})
    if not manifest.get('generated'):
        raise ValueError('generated hashes have not been pinned yet')
    for row in manifest['generated']:
        if key(row) in expected:
            raise ValueError('duplicate fixture pin')
        expected[key(row)] = row['sha256']
    wanted_paths = set()
    for row in record['files']:
        name = row['target'] if 'target' in row else os.fsdecode(bytes.fromhex(row['target_bytes_hex']))
        target = relative(root, name)
        wanted_paths.add(os.fsencode(name))
        if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
            raise ValueError(f'fixture changed: {key(row)}')
        if expected.pop(key(row), None) != row['sha256']:
            raise ValueError(f'fixture pin missing or different: {key(row)}')
    if expected:
        raise ValueError('pins missing from corpus')
    directory = root / 'zz-hostile/many-files'
    names = {p.name for p in directory.iterdir() if p.is_file() and p.stat().st_size == 0}
    if names != {f'{i:06d}.txt' for i in range(manifest['hostile_empty_files'])}:
        raise ValueError('hostile empty-file set differs')
    wanted_paths.update(os.fsencode('zz-hostile/many-files/' + name) for name in names)
    actual_paths = set()
    for directory, folders, files in os.walk(root):
        if Path(directory) == root:
            folders[:] = [folder for folder in folders if folder not in ('.deepkit', '.git')]
        for name in folders + files:
            if (Path(directory)/name).is_symlink():
                raise ValueError('unexpected symlink in corpus')
        for name in files:
            actual_paths.add(os.fsencode(str((Path(directory)/name).relative_to(root))))
    if actual_paths != wanted_paths:
        raise ValueError('unexpected/missing corpus files')
    print(json.dumps({'verified': True, 'regular_files': len(record['files']), 'empty_hostile_files': len(names), 'source_key': source_key}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get('DKC_CORPUS_DIR', '/data/dkc-corpus')).expanduser()
    parser.add_argument('--manifest', type=Path, default=ROOT / 'bench/fixtures-docs.toml')
    parser.add_argument('--output', type=Path, default=base / 'fixtures-docs')
    parser.add_argument('--source-root', type=Path, default=base / 'mdn-content')
    parser.add_argument('--office-python', type=Path, default=Path(os.environ.get('DKC_FIXTURE_OFFICE_PYTHON', str(Path.home()/'.cache/dkc-fixture-tools/office-v1/bin/python'))))
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--seed-git', action='store_true',
                        help='verify an existing generated tree and add its empty Git seed')
    args = parser.parse_args()
    manifest = tomllib.loads(args.manifest.read_text())
    source_key = hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k != 'generated'}, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    if args.verify or args.seed_git:
        verify(output, manifest, source_key)
        if args.seed_git:
            seed_git(output)
        verify_git(output)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError('output is not empty; verify it or choose a new output path')
    output.mkdir(parents=True, exist_ok=True)
    scratch = output / '.deepkit/build-scratch'; scratch.mkdir(parents=True)
    if command(['git', '-C', args.source_root, 'rev-parse', 'HEAD'], text=True).stdout.strip() != manifest['mdn_rev']:
        raise ValueError('MDN checkout does not match its pin')
    records = []
    def record(target, kind, generated=False):
        records.append({'target': str(target.relative_to(output)), 'kind': kind, 'bytes': target.stat().st_size,
                        'sha256': digest(target), 'generated': generated})
    cache = Path.home()/'.cache/dkc-fixture-sources/arxiv'; cache.mkdir(parents=True, exist_ok=True)
    for paper in manifest['paper']:
        if not re.fullmatch(r'\d{4}\.\d{4,5}v\d+', paper['id']):
            raise ValueError('invalid paper ID')
        cached = cache / (paper['id'] + '.pdf')
        if not cached.exists():
            request = urllib.request.Request(paper['url'], headers={'User-Agent':'dkc/0.1.0 corpus-fixture-preparation'})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(paper['bytes'] + 1)
            if len(data) != paper['bytes'] or hashlib.sha256(data).hexdigest() != paper['sha256']:
                raise ValueError('download differs from its PDF pin')
            put(cached, data); time.sleep(3)
        if cached.stat().st_size != paper['bytes'] or digest(cached) != paper['sha256']:
            raise ValueError('cached PDF differs from its pin')
        target = output / '00-digital' / cached.name
        put(target, cached.read_bytes()); record(target, 'digital_pdf')
    sources = [excerpt(args.source_root, row, manifest['mdn_rev']) for row in manifest['markdown']]
    sections = []
    for index, source in enumerate(sources, 1):
        markdown = f'# {source["title"]}\n\n{source["text"]}\n'
        rendered = command(['pandoc', '-f', 'commonmark-raw_html', '-t', 'html', '--wrap=none'], input=markdown.encode()).stdout.decode()
        sections.append(f'<section><small>MDN manual · {index}/300</small>{rendered}<p class="credit">{html.escape(source["url"])}</p></section>')
    css = '@page{size:A4;margin:20mm}body{font:11pt sans-serif}h1{font-size:18pt}section{break-after:page;overflow-wrap:anywhere}section:last-child{break-after:auto}.credit{font-size:8pt}pre{white-space:pre-wrap}'
    manual = output / '00-digital/manual-300-pages.pdf'
    render_html('<!doctype html><meta charset="utf-8"><style>'+css+'</style>'+''.join(sections), manual, 300, scratch)
    record(manual, 'digital_pdf', True)
    for index in range(1, 6):
        target = output / f'00-digital/invoice-{index:02d}.pdf'
        text = f'<!doctype html><meta charset="utf-8"><h1>Fixture invoice {index:02d}</h1><p>Date: 2020-01-01</p><p>Customer: Example document team</p><table><tr><th>Service</th><th>EUR</th></tr><tr><td>Document preparation {index}</td><td>{index*100}.00</td></tr></table><p>Public synthetic invoice. No payment is due.</p>'
        render_html(text, target, 1, scratch); record(target, 'digital_pdf', True)
    for paper in (p for p in manifest['paper'] if p['scan']):
        import pikepdf
        original = output/'00-digital'/(paper['id']+'.pdf')
        with pikepdf.open(original) as pdf:
            pages = len(pdf.pages)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            work = Path(directory)
            command(['gs','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=jpeg','-r200','-dJPEGQ=85',f'-sOutputFile={work}/page-%04d.jpg',original])
            images = sorted(work.glob('page-*.jpg'))
            if len(images) != pages:
                raise ValueError('scan rasterization page count differs')
            raw = work/'scan.pdf'
            command(['img2pdf','--engine','internal','--nodate','--output',raw,*images])
            target = output/'01-scanned'/(paper['id']+'.pdf')
            normalized_pdf(raw, target, pages); record(target, 'scanned_pdf', True)
    office_inputs = scratch/'office-inputs.json'; office_inputs.write_text(json.dumps(sources[30:50]))
    command([args.office_python,'-B',ROOT/'scripts/corpus/make-docs-office.py','--inputs',office_inputs,'--output',output/'02-office'])
    for target in sorted((output/'02-office').iterdir()):
        record(target, 'office', True)
    for image in manifest['image']:
        source = relative(args.source_root, image['source'])
        if digest(source) != image['sha256']:
            raise ValueError('image source differs from its pin')
        target = relative(output, image['target'])
        put(target, source.read_bytes()); record(target, 'image')
    # Valid small WebP plus an unknown RIFF chunk: exactly 8 MiB of container
    # bytes, independently exercising the image-byte cap without a giant decode.
    from PIL import Image
    encoded = BytesIO(); Image.new('RGB',(32,32),(40,100,160)).save(encoded,format='WEBP',lossless=True)
    data = bytearray(encoded.getvalue()); padding = 8*1024*1024-len(data)-8
    data.extend(b'JUNK'+struct.pack('<I',padding)+bytes(padding)); data[4:8]=struct.pack('<I',len(data)-8)
    with Image.open(BytesIO(data)) as image:
        image.load(); assert image.size == (32,32)
    target = output/'03-images/oversized-container.webp'; put(target,data); record(target,'oversized_image',True)
    text = '# Document fixture corpus\n\nPinned extraction and enrichment inputs. Generated files are synthetic.\n\nMDN excerpts and images: https://github.com/mdn/content/tree/'+manifest['mdn_rev']+'\nMDN prose attribution/license: https://github.com/mdn/content/blob/'+manifest['mdn_rev']+'/LICENSE.md (CC-BY-SA 2.5). Paper URLs are recorded in the source manifest.\n\n'
    for image in manifest['image']:
        if image['captioned']:
            caption = Path(image['source']).stem.replace('-',' ').replace('_',' ')
            text += f'![MDN documentation illustration: {caption}]({image["target"]})\n\nSource: https://github.com/mdn/content/blob/{manifest["mdn_rev"]}/{image["source"]}\n\n'
    target = output/'README.md'; put(target,text.encode()); record(target,'readme',True)
    hostile = output/'zz-hostile'; hostile.mkdir()
    target = hostile/'large.json'
    with target.open('wb') as stream:
        prefix=b'{"payload":"'; stream.write(prefix)
        remaining=200*1024*1024-len(prefix)-2
        while remaining:
            count=min(remaining,1024*1024); stream.write(b'a'*count); remaining-=count
        stream.write(b'"}')
    record(target,'hostile_large_json',True)
    target=hostile/'sparse.bin'
    with target.open('wb') as stream: stream.truncate(5*1024*1024*1024)
    record(target,'hostile_sparse_binary',True)
    target=hostile/'empty.txt'; put(target,b''); record(target,'hostile_empty',True)
    bad=os.fsencode(hostile)+b'/invalid-\xff.txt'
    invalid_text = b'Invalid UTF-8 filename fixture.\n'
    with open(bad,'wb') as stream: stream.write(invalid_text)
    records.append({'target_bytes_hex':b'zz-hostile/invalid-\xff.txt'.hex(),'kind':'hostile_invalid_utf8','bytes':os.stat(bad).st_size,'sha256':hashlib.sha256(invalid_text).hexdigest(),'generated':True})
    many=hostile/'many-files'; many.mkdir()
    for index in range(manifest['hostile_empty_files']): (many/f'{index:06d}.txt').touch()
    shutil.rmtree(scratch)
    result={'source_key':source_key,'files':records,'empty_hostile_files':manifest['hostile_empty_files'],'generated_pins_present':bool(manifest.get('generated'))}
    (output/'.deepkit/fixture-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    seed_git(output)
    print(json.dumps({'output':str(output),'regular_files':len(records),'empty_hostile_files':manifest['hostile_empty_files'],'generated_pins_present':result['generated_pins_present']}))
    if manifest.get('generated'):
        verify(output, manifest, source_key)


if __name__ == '__main__':
    main()
