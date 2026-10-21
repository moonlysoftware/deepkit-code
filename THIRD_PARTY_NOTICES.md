# Third-party notices

Deepkit Code combines Rust packages, native libraries and language grammars.
They retain their own copyright and license terms. Deepkit Code's original
code is licensed separately as described in [LICENSE.md](LICENSE.md).

Release archives include `third-party-licenses.tar.gz` (extract with `tar -xzf third-party-licenses.tar.gz`), containing full license and
notice texts plus a SHA-256 manifest. The bundle inventories the complete
`Cargo.lock`, including optional, platform, build and test dependencies. It is
a conservative inventory, not a claim that every listed package is linked into
every binary. Package versions and crate archive checksums identify the sources.

## Native code and copied sources

| Component | Provenance | License / notices |
| --- | --- | --- |
| sqlite-vector 1.1.0 | [Apache revision `cfe78ae17c108938b3c87713775b94f7ffe24b87`](https://github.com/sqliteai/sqlite-vector/tree/cfe78ae17c108938b3c87713775b94f7ffe24b87) | Apache-2.0; upstream license retained. DKC modifications are documented in `crates/sqlite-vector-sys/README.md` and marked in the modified C files. |
| FP16 headers used by sqlite-vector | [Revision `98b0a46bce017382a6351a19577ec43a715b6835`](https://github.com/Maratyszcza/FP16/tree/98b0a46bce017382a6351a19577ec43a715b6835) | MIT; Copyright 2017 Facebook Inc. and Georgia Institute of Technology. All three vendored headers match this revision byte for byte. |
| SQLite 3.53.2 | `libsqlite3-sys` 0.38.2 bundled amalgamation | SQLite's deliverable library is [public domain](https://www.sqlite.org/copyright.html). The Rust wrapper has its own MIT license, retained in the bundle. |
| llama.cpp / ggml | [Revision `e79e4bf660e19f2ad851e06c6913f7a8c5852621`](https://github.com/ggml-org/llama.cpp/tree/e79e4bf660e19f2ad851e06c6913f7a8c5852621), bundled by `llama-cpp-sys-2` 0.1.156 | MIT, Copyright 2023–2026 The ggml authors. Native source attribution lines and third-party notices are also retained. Rust wrappers offer MIT OR Apache-2.0. |
| nlohmann/json, stb_image, cpp-httplib and other included native material | Pinned llama.cpp source above | Their original notices are retained in the native and crate portions of the bundle. Some optional native components are not used by DKC's release targets. |
| gix-index 0.55.0 | Local patch of upstream revision `d2af4ed5532ea660fbd643e48d8925cd88de5ee0` | MIT OR Apache-2.0. Both texts and `vendor/gix-index/PATCH.md` are retained. |
| Tree-sitter parsers and queries | Crate versions/checksums in `Cargo.lock`; copied query provenance in `src/treat/grammar_queries/README.md` | Grammar licenses are included per package. Copied C, C++, C#, Scala and OCaml-interface MIT query notices are additionally included under `grammar-queries/`. |

The sqlite-vector Apache revision has identical upstream source/header and API
bytes to DKC's previous pin. Its adoption changes licensing provenance while
preserving the existing local runtime patches. The independent MIT terms of
FP16 still apply. The Apache license text SHA-256 is
`421717de661f6707b9829bd7d459994af9fc5c7b400335da822a817ffd8a42a1`.

The registry bundle preserves compound-license notices, including `ring`'s
Apache/ISC/BoringSSL notices, Unicode data notices, and the `NOTICE` files of
PDFOxide and OfficeOxide. Package names and marks remain their owners' property.
No package in the reviewed lockfile declares MPL-2.0; its presence in the
repository's permitted-license list does not mean it is used.

## Models downloaded separately

Model weights are not included in the release archives or Homebrew package.
When a user enables local embeddings or reranking, DKC downloads files pinned
by revision, byte size and SHA-256 in `src/models/models.toml`.

- [Qwen3-Embedding-0.6B GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/tree/370f27d7550e0def9b39c1f16d3fbaa13aa67728)
  declares Apache-2.0 in its pinned model card.
- [Deepkit Code's Qwen3-Reranker-0.6B GGUF conversions](https://huggingface.co/bartscheffer/deepkit-code-reranker-gguf/tree/90736d4de467cd106d46f5b576f9e59255c43a51)
  declare Apache-2.0 and retain a license, model card and artifact manifest.
  Their source model is Qwen/Qwen3-Reranker-0.6B.

The GitHub organization transfer does not change these pinned Hugging Face
addresses. Redistributing or converting weights separately requires retaining
those model terms and provenance; the application's dual license does not
replace them.

## Release maintainers

Generate the bundle offline from checksum-verified `.crate` archives and reviewed
pinned notice supplements. Fail on missing license texts or an unreviewed source
obligation; do not replace a missing copyright notice with a guessed author.
Include `LICENSE.md`, `LICENSE-MIT`, `LICENSE-APACHE`, this file and the full bundle
in each actual downloadable archive. Check archive contents and installed
Homebrew `pkgshare` notices before publishing. The manifest covers attribution;
`cargo deny check` remains a separate dependency-license and advisory gate.
