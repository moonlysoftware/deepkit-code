# T2.10 grammar queries

The registry in `../languages.rs` statically links all 30 planned languages,
including separate TSX, OCaml interface and Markdown inline parsers (33 variants).
Versions are pinned in Cargo.toml and Cargo.lock. Config and Markdown retain their
specified config/section chunkers and do not create code graph symbols or edges.
SCSS/Sass and literate Haskell retain bounded line fallback; these are not claims
of dedicated dialect grammar coverage.

Minimal queries were written for Kotlin, Bash, HTML, CSS, SQL, Zig, Haskell,
Protobuf, JSON, YAML, TOML, Markdown block/inline and Dockerfile. They intentionally
provide approximate symbols/references, not full compiler name resolution.

Copied MIT queries retain accompanying `<name>.LICENSE` notices:

- C: tree-sitter-c 0.24.2. Function definitions include bodies; direct calls and
  one/two-pointer return declarators are captured.
- C++: tree-sitter-cpp 0.23.4. Function and qualified-method definitions include
  bodies; direct calls, pointer and reference returns are captured.
- C#: tree-sitter-c-sharp 0.23.5. Removed an invalid duplicate `@module` capture
  and added direct invocation references.
- Scala: tree-sitter-scala 0.26.2, unchanged.
- OCaml interface: tree-sitter-ocaml 0.25.0 plus `val` declaration capture.

Other queries come from the pinned crate's exported TAGS_QUERY. Archive URLs and
checksums are recorded in `src/treat/grammar_queries/sources.json`. C++'s license
comes from its v0.23.4 upstream tag because the crate archive omits the notice.

`fixtures/lang/remaining.json` maps the new variants to permanent inputs and
reviewed `production.json` goldens. Production tests check routing, unit kinds,
text, ranges, graph names/edges, deterministic output and zero parse failures;
the existing eight-language goldens protect their previous chunk partitions.
Additional generated tests cover oversized containers/leaves and admission caps.
