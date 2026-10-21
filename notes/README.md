# Local development records

This directory is for local run logs, journals and resumable work. Those files
are ignored by Git. The tracked STATE and QUESTIONS files provide a compact
handoff; user-facing guides and selected evidence are in [docs](../docs/README.md)
and [bench/results](../bench/results/beta-20261014/).

Before the beta cleanup, the complete development record was archived with a
verified Git bundle and working-tree patch. Earlier Git history is preserved in
`moonlysoftware/deepkit-code-history`; the current repository starts fresh on
21 October 2026. Restoring
historical files is a maintainer operation, not an installation prerequisite.

Long local commands may still use `scripts/run-detached.sh`: read the matching
log and terminal exit code before resuming. `notes/STOP`, when present, remains
a local instruction against unattended corpus/goal work; never stage or remove
it automatically.

## Archived specifications and reference documents

The owner approved archiving the internal plan, original specifications and
Deepkit reference documents on 14 October 2026. Source comments such as
`v1.md §7` refer to the preserved specification at
`a7e45b06ce1c40c7648c44359bb248c83f7b232b`. Read a historical document from a separate history clone without
adding the old history to the current repository:

```sh
git clone git@github.com:moonlysoftware/deepkit-code-history.git ../deepkit-code-history
git -C ../deepkit-code-history show a7e45b06ce1c40c7648c44359bb248c83f7b232b:v1.md
git -C ../deepkit-code-history show a7e45b06ce1c40c7648c44359bb248c83f7b232b:plan/phases/phase-4-corpus-bench-ship.md
```

The verified external archive is recorded in STATE.md. The history repository
retains the original commits unchanged. Historical receipt URLs naming the old
`moonlysoftware/deepkit-code` repository must now use `deepkit-code-history`. This cleanup does not complete or waive the historical v1 acceptance
criteria. The stable-release checker must still refuse release when its
archived specifications, report or supporting evidence are absent. A future
v1 qualification task must restore and reconcile those inputs before using it.
