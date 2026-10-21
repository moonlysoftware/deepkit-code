# Decisions and remaining owner actions

The active owner checklist is [the beta handover](../docs/beta-release.md).
The owner transferred the source repository to `moonlysoftware/deepkit-code`;
Git/SSH/admin access and private visibility are verified. The next instruction
authorized preparation of a private `moonlysoftware/homebrew-tap`. Public
visibility, a release/tag and registry publication remain pending. The owner
then requested updating main; PR #3 merged as `e1a1777` after its final checks.

## Accepted beta review and cleanup

On 14 October 2026 the owner approved the supplied checklist item:
“Review PR #3, retained history and the protected-document cleanup proposal.”
The approved patch archives 33 internal plan/spec/reference files and replaces
AGENTS.md with the prepared contributor guide. Git history is retained unchanged.
The patch SHA-256 is
`115224bc124e0a6ef84d456f1dc6c2c0b01a9e003b63cfaa433284561dda2046`;
its backup and restoration instructions are recorded in STATE.md and README.md.
That approval covered review/cleanup. The later owner transfer and private-tap
request extended the scope to private setup and testing. The subsequent request
to update main authorized source integration, now complete with retained history.
Public visibility, release/tag and public tap publication remain owner actions.
Actions logs/artifacts still need their final visibility review before publication.

## Remaining beta decisions and actions

- Initial recommendation: Apple Silicon first. Native Homebrew passed on ARM
  Mac and a Linux GNU diagnostic; Intel Mac and portable Linux packages still
  need their own evidence. Windows remains unsupported.

## Preserved accepted decisions

- Cost rates stay empty/null; testing has no monetary cap. Provider deadlines,
  request/rate limits, privacy and storage/work caps still apply.
- The distribution size cap is 50 MB per complete compressed archive, not the
  installed executable (PCR-026, accepted September 25).
- Aurora qualification is deferred beyond v1; remaining matrix scope is 26
  targets/81 lanes (PCR-032). Deferral is not a pass.
- Ripgrep goldens are reviewed. Laravel/Django/Hugo goldens remain drafts; a
  review sheet does not approve their labels.
- Immutable Moonly weight revision qualification was deferred. Deployment IDs
  and live drift checks remain required for formal acceptance; unchanged IDs
  do not prove unchanged weights.
- v1 corpus, performance, native Metal/installer checks and PCR dispositions
  remain open. Preserve known misses; do not reinterpret beta as v1 completion.

The earlier detailed questions, answers and PCRs are in the verified external
archive and historical Git at `a7e45b06ce1c40c7648c44359bb248c83f7b232b`.
