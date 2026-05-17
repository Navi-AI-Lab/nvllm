# Qwen3.6 NVFP4 — tokenizer_config patch

Every Qwen3.6-27B NVFP4 checkpoint observed so far (Unsloth, Huihui) ships a
tokenizer_config.json that declares the tokenizer as a transformers 5.x wrapper:

```json
{
  "backend": "tokenizers",
  "tokenizer_class": "TokenizersBackend"
}
```

The `nvllm:gb10` image pins `transformers==4.57.6` (build constraint —
`transformers>=5` breaks the docker build). 4.57.6 has no `TokenizersBackend`
symbol, so model load fails with:

```
ValueError: Tokenizer class TokenizersBackend does not exist or is not
currently imported.
```

Same root cause as
[vllm-project/vllm#38024](https://github.com/vllm-project/vllm/issues/38024).

### What this patch does

Substitutes the wrapper hint with the real underlying class:

```json
{
  "tokenizer_class": "Qwen2Tokenizer"
  // "backend" key removed
}
```

The underlying `tokenizer.json` (the actual vocab + merges) is unchanged —
only the loader-class metadata is swapped so transformers 4.57.6 can resolve
a concrete class. All special tokens, chat template, and model-specific
multimodal tokens remain intact.

### How it's consumed

`scripts/serve-qwen36.sh` auto-detects checkpoints carrying the bad wrapper
and bind-mounts this file *over* the snapshot's `tokenizer_config.json`
inside the container:

```bash
HOST  : models/qwen36-tokenizer-patch/tokenizer_config.json
GUEST : /root/.cache/huggingface/hub/models--<repo>/snapshots/<id>/tokenizer_config.json
```

The snapshot path is derived from `$HF_MODEL` (replace `/` with `--`). The
HuggingFace cache on the host is **not modified**; a future `hf download`
will not clobber the patch, and removing the bind-mount restores the
upstream config.

Disable with `NVLLM_TOKENIZER_PATCH=0` if/when the image moves to
transformers 5.x, or if you want to see the raw upstream failure.

### Regenerating

```bash
SNAP=~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/*/
python3 -c "
import json
cfg = json.load(open('$SNAP/tokenizer_config.json'))
cfg['tokenizer_class'] = 'Qwen2Tokenizer'
cfg.pop('backend', None)
json.dump(cfg, open('models/qwen36-tokenizer-patch/tokenizer_config.json', 'w'), indent=2)
"
```
