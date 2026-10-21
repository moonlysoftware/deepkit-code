# Document corpus preparation

`fetch.sh --only mdn-content` provisions the full pinned MDN repository from
`bench/corpus.toml`. `fetch.sh --verify --only mdn-content` verifies it without
moving an existing checkout. `DKC_CORPUS_DIR` defaults to `/data/dkc-corpus`.

`make-fixtures-docs.sh` builds the document corpus from
`bench/fixtures-docs.toml`. Downloads use explicit arXiv versions and SHA-256
pins. MDN images and Markdown inputs have commit/path/content pins. The generated
manual, invoices, scans, Office files and hostile files also have observed hashes.
The builder refuses a nonempty output directory, checks all pins at completion,
and seeds an empty local Git commit. The seed gives `dkc bench run` a valid
`HEAD` without tracking the 5 GiB sparse file or 100,000 hostile empty files.

```sh
scripts/corpus/fetch.sh --only mdn-content
scripts/corpus/make-fixtures-docs.sh
scripts/corpus/make-fixtures-docs.sh --verify
```

Use `--output PATH` for another independent build, and `--source-root PATH` for
the MDN checkout. Verification checks every expected file, rejects extra files
and symlinks, checks the complete 100,000-file directory, and verifies the
empty Git seed. To repair an older pinned build that lacks Git, run
`scripts/corpus/make-fixtures-docs.sh --seed-git`; it verifies the fixture
bytes first. The `.deepkit/` directory holds preparation records and the
later index. Neither `.git/` nor `.deepkit/` is part of the file-content pin.

Generation uses Ghostscript, img2pdf, Pandoc, Chromium, pikepdf and Pillow. The
historical tool-version receipt is retained in the development archive under
`notes/bench/t3-9-tool-versions-1.json`; see [local records](../../notes/README.md).
Install the Office
libraries with `python -m pip install -r scripts/corpus/requirements-docs-office.lock`
in a virtual environment, then pass its Python executable with `--office-python`
or `DKC_FIXTURE_OFFICE_PYTHON`. The default points to the existing bench-machine
fixture environment. Generated-byte verification detects rendering/library drift.

The 50 digital PDFs include 44 papers, the 300-page MDN manual and five synthetic
invoices. Ten papers also have complete 200-dpi JPEG scans wrapped as image-only
PDFs. The Office set contains 7 DOCX, 7 XLSX and 6 PPTX; one DOCX is deliberately
encrypted with public fixture-only parameters, and one XLSX is truncated. Those
two are expected L0 refusals. The 100 collected images have 20 README captions.
A separate valid WebP has an 8 MiB padded container to exercise the image-byte cap.

For the document measurement, copy `bench/profiles/fixtures-docs.toml` to the
corpus's `.deepkit/config.toml`. It selects Moonly routing and disables Qwen
thinking, while setting the planned 50,000-file cap and a 64 MiB ordinary-file cap
for real papers/scans. The ordinary cap still rejects the 200 MiB JSON and 5 GiB
sparse binary; the default 5 MiB image cap rejects the oversized WebP. Provider
credentials remain in the configured environment variable or user credential
file. Fixture verification alone does not establish indexing or benchmark
acceptance; see the [retained measurements](../../docs/benchmarks.md).
