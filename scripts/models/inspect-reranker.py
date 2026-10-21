#!/usr/bin/env python3
"""Validate converted reranker metadata and classifier rows against the pinned HF source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('llama_source', type=Path)
    parser.add_argument('model_source', type=Path)
    parser.add_argument('artifact', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.llama_source / 'gguf-py'))
    import gguf
    import numpy as np
    from safetensors import safe_open
    from transformers import AutoTokenizer

    reader = gguf.GGUFReader(args.artifact)
    keys = ('general.architecture', 'general.file_type', 'qwen3.pooling_type',
            'qwen3.classifier.output_labels', 'tokenizer.chat_template.rerank')
    metadata = {key: reader.get_field(key).contents() for key in keys}
    assert metadata['general.architecture'] == 'qwen3'
    assert metadata['qwen3.pooling_type'] == int(gguf.PoolingType.RANK)
    assert metadata['qwen3.classifier.output_labels'] == ['yes', 'no']
    template = metadata['tokenizer.chat_template.rerank']
    assert '<Query>: {query}' in template and '<Document>: {document}' in template
    assert template.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    classifier = next(t for t in reader.tensors if t.name == 'cls.output.weight')
    assert list(classifier.shape) == [1024, 2]
    actual = gguf.quants.dequantize(classifier.data, classifier.tensor_type).reshape(2, 1024)
    tokenizer = AutoTokenizer.from_pretrained(args.model_source, local_files_only=True)
    ids = [tokenizer.convert_tokens_to_ids(label) for label in ('yes', 'no')]
    with safe_open(args.model_source / 'model.safetensors', framework='pt') as source:
        weights = source.get_slice('model.embed_tokens.weight')
        expected = np.stack([weights[index:index + 1].float().numpy()[0] for index in ids])
    cosine = np.sum(actual * expected, axis=1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1))
    assert np.isfinite(actual).all() and (cosine > 0.999).all()
    with args.artifact.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    print(json.dumps({'file': args.artifact.name, 'size': args.artifact.stat().st_size,
                      'sha256': sha, 'metadata': metadata, 'tensor_count': len(reader.tensors),
                      'classifier_type': classifier.tensor_type.name, 'classifier_token_ids': ids,
                      'classifier_row_cosines': cosine.tolist(),
                      'classifier_max_abs_error': float(np.max(np.abs(actual - expected)))}, indent=2))


if __name__ == '__main__':
    main()
