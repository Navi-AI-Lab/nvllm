# Qwen3.6 MTP NVFP4 — tokenizer_config patch

The published checkpoint
[`sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP`](https://huggingface.co/sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP)
declares its tokenizer as a transformers 5.x wrapper:

```json
{
  "backend": "tokenizers",
  "tokenizer_class": "TokenizersBackend"
}
```

The `nvllm:gb10` image pins `transformers==4.57.6` (build constraint —
`transformers>=5` breaks the docker build). 4.57.6 has no
`TokenizersBackend` symbol, so model load fails with:

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

The underlying `tokenizer.json` (the actual vocab + merges) is unchanged
— only the loader-class metadata is swapped so transformers 4.57.6 can
resolve a concrete class. All special tokens, chat template, and
model-specific multimodal tokens remain intact.

### How it's consumed

`scripts/serve-cute-mtp.sh` bind-mounts this file *over* the snapshot's
`tokenizer_config.json` inside the container:

```bash
HOST  : models/qwen3.6-mtp-tokenizer-patch/tokenizer_config.json
GUEST : /root/.cache/huggingface/hub/models--sakamakismile--Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP/snapshots/<id>/tokenizer_config.json
```

The HuggingFace cache on the host is **not modified**; a future `hf
download` will not clobber the patch, and removing the bind-mount
restores the upstream config.

Disable with `NVLLM_TOKENIZER_PATCH=0` if/when the image moves to
transformers 5.x.

### Regenerating

```bash
SNAP=~/.cache/huggingface/hub/models--sakamakismile--Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP/snapshots/*/
python3 -c "
import json
cfg = json.load(open('$SNAP/tokenizer_config.json'))
cfg['tokenizer_class'] = 'Qwen2Tokenizer'
cfg.pop('backend', None)
json.dump(cfg, open('models/qwen3.6-mtp-tokenizer-patch/tokenizer_config.json', 'w'), indent=2)
"
```
