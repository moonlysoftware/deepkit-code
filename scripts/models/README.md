# Preparing the local reranker artifacts

These developer tools prepare the Qwen3 reranker files for the embedded registry.
They are separate from `dkc models pull`. Run them in a build directory outside
both this repository and the user's model cache. Nothing here uploads an artifact.

The inputs are pinned in `reranker-source.json`: the official Qwen revision and
its Git-blob/LFS hashes, plus the complete llama.cpp revision used by our pinned
Rust binding. `prepare-reranker.py` verifies every downloaded file. It reuses valid
complete files and refuses an invalid existing file; interrupted `.part` downloads
restart. This preparation helper does not replace the product's resumable downloader.

With Python 3.12 and a virtual environment available:

```sh
# Choose an empty build directory outside the repository/model cache.
build_dir="$HOME/.cache/dkc-artifact-builds"
mkdir -p "$build_dir"
git init "$build_dir/llama-e79e4bf"
git -C "$build_dir/llama-e79e4bf" -c http.userAgent=dkc/0.1.0 fetch --depth 1 \
  https://github.com/ggml-org/llama.cpp.git e79e4bf660e19f2ad851e06c6913f7a8c5852621
git -C "$build_dir/llama-e79e4bf" checkout --detach FETCH_HEAD
python3.12 -m venv "$build_dir/converter-venv"
"$build_dir/converter-venv/bin/pip" install -r scripts/models/converter-requirements.lock
python3 scripts/models/prepare-reranker.py "$build_dir/Qwen3-Reranker-0.6B"
mkdir -p "$build_dir/artifacts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$build_dir/converter-venv/bin/python" "$build_dir/llama-e79e4bf/convert_hf_to_gguf.py" \
  "$build_dir/Qwen3-Reranker-0.6B" --outtype q8_0 \
  --outfile "$build_dir/artifacts/Qwen3-Reranker-0.6B-Q8_0.gguf"
```

Repeat conversion with `--outtype f16` and an output filename ending in `-f16.gguf`.
The requirements lock records the environment used for the prepared artifacts;
llama.cpp's own `gguf-py` is selected by both conversion and inspection.

Inspect each artifact and run the real Rust CPU smoke test:

```sh
artifact="$build_dir/artifacts/Qwen3-Reranker-0.6B-Q8_0.gguf"
"$build_dir/converter-venv/bin/python" scripts/models/inspect-reranker.py \
  "$build_dir/llama-e79e4bf" "$build_dir/Qwen3-Reranker-0.6B" "$artifact"
DKC_RERANKER_GGUF="$artifact" cargo test --bin dkc \
  models::llama::tests::reranker_artifact_sanity -- --ignored --nocapture
```

Inspection checks `cls.output.weight`, its shape and row correspondence to the
original yes/no token weights, Rank pooling, labels, the rerank template, size and
SHA-256. The Rust test checks actual three-pair scores. This pinned native Qwen3
Rank graph already applies softmax; consumers must not apply it again.

Use `scripts/run-detached.sh` for downloads/conversion that must survive a session.
Keep commands, environment and results in the local, ignored `notes/` directory;
see [local records](../../notes/README.md). Before adding registry URLs,
obtain the project-owned hosting repository, publish the validated bytes, then
verify the hosted revision, size and hash. These preparation checks cover the
model artifacts; product reranking and search fallback have separate tests.

The first validated artifacts are published at
https://huggingface.co/bartscheffer/deepkit-code-reranker-gguf, revision
`90736d4de467cd106d46f5b576f9e59255c43a51`. The product registry uses these
immutable URLs. The historical anonymous full-byte verification receipt is
retained in the development archive under
`notes/reports/t1-4n-hosted-reranker.json`. Do not rerun the one-off creation script
against this existing repository; future updates need a fresh reviewed artifact
manifest.
