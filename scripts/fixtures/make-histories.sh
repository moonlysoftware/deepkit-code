#!/usr/bin/env bash
# Generate real, deterministic PR styles and expected member sets (v1 §4.4).
set -euo pipefail
exec python3 - "$@" <<'PY'
from pathlib import Path
import json,os,subprocess,sys
if len(sys.argv)<2:
    raise SystemExit('usage: make-histories.sh NEW_OUTPUT_DIRECTORY [STYLE ...]')
out=Path(sys.argv[1]).resolve()
if out.exists(): raise SystemExit('refusing to replace existing fixtures')
out.mkdir(parents=True)
manifest=[]
for style in sys.argv[2:] or ['github-merge','squash','gitlab','plain','rebase','skew','over-cap']:
    root=out/style;root.mkdir()
    env={**os.environ,'GIT_CONFIG_GLOBAL':'/dev/null','GIT_CONFIG_SYSTEM':'/dev/null'}
    def git(*args,date='2026-01-01T12:00:00Z',stdin=None):
        p=subprocess.run(['git','-C',str(root),*args],env={**env,'GIT_AUTHOR_DATE':date,'GIT_COMMITTER_DATE':date},input=stdin,text=True,capture_output=True,check=True)
        return p.stdout.strip()
    git('init','-q','-b','main');git('config','user.name','PR Fixture');git('config','user.email','pr@example.invalid');git('config','commit.gpgSign','false');git('config','core.hooksPath','/dev/null')
    (root/'root.rs').write_text('pub fn initial() {}\n');git('add','.');git('commit','-q','-m','Initial source')
    base=git('rev-parse','HEAD');git('checkout','-q','-b','topic')
    if style=='over-cap':
        tree=git('rev-parse','HEAD^{tree}');head=base
        for n in range(501): head=git('commit-tree',tree,'-p',head,stdin=f'Member {n}\n',date='2026-01-02T12:00:00Z')
        git('update-ref','refs/heads/topic',head);git('reset','--hard','-q',head)
    else:
        for n in [1,2]:
            name='private' if style=='mixed' and n==1 else 'billing'
            (root/(name+'.rs')).write_text(f'pub fn {name}() -> i32 {{\n    {n}\n}}\n')
            git('add','.');git('commit','-q','-m',f'Billing member {n}','-m',f'Body {n}\nAnother line',date=('2030-01-01T12:00:00Z' if style=='skew' else f'2026-01-0{n+1}T12:00:00Z'))
        head=git('rev-parse','HEAD')
    git('checkout','-q','main');(root/'main.rs').write_text('pub fn mainline() {}\n');git('add','.');git('commit','-q','-m','Mainline work',date='2026-01-04T12:00:00Z')
    parent1=git('rev-parse','HEAD')
    if style=='rebase':
        git('checkout','-q','topic');git('rebase','main');git('checkout','-q','main');git('merge','--ff-only','-q','topic');expected=[]
    elif style=='squash':
        git('merge','--squash','-q','topic');git('commit','-q','-m','Billing squash (#102)','-m','Squashed body',date='2026-01-05T12:00:00Z');merge=git('rev-parse','HEAD');expected=[{'merge':merge,'members':[merge],'number':102,'strategy':'squash'}]
    else:
        subject={'gitlab':"Merge branch topic See merge request team/repo!103",'plain':"Merge branch 'topic' into main"}.get(style,'Merge pull request #101 from fixture/topic')
        git('merge','--no-ff','-q','topic','-m',subject,'-m','Billing changes',date='2026-01-05T12:00:00Z');merge=git('rev-parse','HEAD')
        members=git('rev-list',f'{parent1}..{head}').splitlines()
        expected=[] if style=='over-cap' else [{'merge':merge,'members':members,'number':103 if style=='gitlab' else None if style=='plain' else 101,'strategy':{'gitlab':'gitlab','plain':'plain'}.get(style,'merge')}]
    (root/'after.rs').write_text('pub fn uncovered() {}\n');git('add','.');git('commit','-q','-m','Uncovered follow-up',date='2031-01-01T12:00:00Z' if style=='skew' else '2026-01-06T12:00:00Z')
    (root/'.deepkit').mkdir();(root/'.deepkit/config.toml').write_text("[embedder]\nprofile='local'\n[llm]\nenabled=false\n[rerank]\nmode='off'\n")
    manifest.append({'style':style,'root':str(root),'head':git('rev-parse','HEAD'),'prs':expected})
(out/'expected.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out/'expected.json')
PY
