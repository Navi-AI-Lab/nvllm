# Silly Streaming

### GDS Reader + nvCOMP Cascaded Decompression on GB10

> A.k.a. "the MacFan move, but with a decompression engine." Research note, no implementation.

**Date:** 2026-04-19
**Target hardware:** NVIDIA DGX Spark / GB10 (SM120/SM121), 128 GB unified LPDDR5X
**Question:** can we use cuFile + nvCOMP + Blackwell's hardware Decompression Engine to do just-in-time weight streaming from NVMe, and in doing so run frontier MoE models larger than unified memory on a single Spark?

This is a feasibility study, not a design doc. No code is proposed. The goal is to figure out whether the idea survives contact with the hardware math, name the one place where it's genuinely interesting, and note the experiments that would confirm or kill it.

---

## TL;DR

Three findings, in the order they matter:

1. **The seam exists.** nvcomp 4.2.0 explicitly added Blackwell Hardware Decompress Engine (HW-DE) support for Snappy, Gzip, and Deflate. All the plumbing — libcufile, gds-tools, Python bindings — is already installed on Spark. The one unverified link in the chain is whether the DE block is actually fused and enabled on GB10 (SM120) versus datacenter-only on SM100 (B100/B200). That's a one-microbench question, not a multi-week question.

2. **On unified memory, cuFile/GDS is *not* what makes this interesting.** Classic GDS matters because it bypasses the CPU bounce buffer on dGPU + NVMe systems. On GB10, GPU memory *is* system memory — the bounce is already zero. What cuFile still buys you is async I/O queue depth, page-cache bypass, and a sane batching story. That's worth having but it's an I/O-path optimization, not the thing that makes the project exciting.

3. **The real win is MoE expert streaming, not whole-model streaming.** For dense ≤120 GB models you can just fit them. For dense >120 GB models the bandwidth math doesn't pencil out for per-token decode even with HW decomp. But for *sparse* MoE — where each token only activates a small fraction of experts — compressed SSD-resident experts + HW-DE-driven prefetch sits in the range where it could actually work, because you're streaming a ~GB of experts per layer, not the whole model per token.

**Why this is newly timely (verified April 19 2026):** the frontier has converged hard on ~1T-total MoE with 3-5% active-parameter fractions. Kimi K2.5 (Jan 27), GLM-5 (Feb 11), GLM-5.1 open weights (Apr 7), Kimi K2.6 Code Preview (Apr 13), DeepSeek V4 (April 2026 launch window) — five releases in three months, all ~700B-1T total, all 32-44B active. At NVFP4 these sit in the 350-520 GB range. None fit in GB10's 120 GB usable unified memory. Streaming from NVMe isn't a clever optimization for these — it's the only way they run on this hardware at all. See §3.1 for the target-model table with citations.

Everything below is the evidence for those three claims.

---

## Part I — Feasibility recon

### 1.1 Hardware audit

#### What's actually on the box

From a fresh `nvidia-smi` and friends (verified on this host, 2026-04-19):

- **GPU:** NVIDIA GB10, driver 590.48.01, CUDA 13.1 runtime, CUDA 13.2 toolkit
- **Memory:** "Not Supported" under Memory-Usage — the standard unified-memory tell; `/proc/meminfo` reports `MemTotal: 125 GB` (128 GB hardware minus firmware/driver reservations)
- **Storage:** 1 × 1 TB NVMe, `ESL01TBTLCZ-27J2-TYN`, PCIe Gen4 × 4 link (`16.0 GT/s PCIe`, width 4)
- **Host kernel:** 6.17.0-1014-nvidia, aarch64 (Grace CPU)

#### cuFile and GDS are first-class on this box

Both CUDA 13.0 and 13.2 toolkits ship with full cuFile support. Installed packages:

```
libcufile-13-2                1.17.0.44-1   arm64
libcufile-dev-13-2            1.17.0.44-1   arm64
gds-tools-13-2                1.17.0.44-1   arm64
nvidia-cufile (pip)           1.15.1.6
/usr/local/cuda/gds/tools/gdsio v1.12
```

That's cuFile library, headers, command-line `gdsio` for microbenchmarks, and Python bindings. Nothing to install.

#### nvCOMP ships a Blackwell HW-DE fast path

From the nvcomp 4.2.0 release notes ([docs.nvidia.com/cuda/nvcomp/release_notes](https://docs.nvidia.com/cuda/nvcomp/release_notes)):

> Added support for Blackwell HW Decompress Engine for Snappy, Gzip, and Deflate

Three things to note:

- **LZ4 and Bitcomp are not on the HW-DE list.** LZ4 in particular runs on the SMs. If you want HW acceleration, pick Deflate/Gzip or Snappy.
- **The API is transparent.** `nvcompBatchedDeflateDecompressAsync` does not expose an HW/SW selector — the library picks the best backend at runtime for the compute capability.
- **Alignment requirements are exposed** via `nvcompBatchedDeflateDecompressRequiredAlignments`. Worth tracking because cuFile also has alignment constraints; the product of the two sets your actual minimum chunk size.

#### What we still don't know

**Is the DE block actually fused on GB10 (SM120)?** Blackwell is a microarchitecture family, not a single die. B100/B200 (SM100) are datacenter parts. GB10 is the consumer/workstation Grace-Blackwell. NVIDIA doesn't publish a per-SKU "is the data decompression engine present" matrix that I could find in an hour. The cleanest answer is a microbench: compress a 1 GB buffer as Deflate, run `nvcompBatchedDeflateDecompressAsync` on Spark and on a known-SM100 node, compare achieved decompression GB/s and SM occupancy. If SM120 achieves the published ~60 GB/s HW-DE Deflate numbers *with low SM utilization*, the DE is present and active. If it matches SM-only numbers or the SMs light up, the DE isn't exposed and you're back to software decomp on the SMs.

This is the one remaining gatekeeper experiment. It's a ~1-hour exercise once you've got a compressed buffer handy.

### 1.2 Bandwidth math

This is where feasibility lives or dies. Four numbers:

| Thing | Number | Source |
|---|---|---|
| NVMe theoretical peak (Gen4 × 4) | ~7.88 GB/s | `16 GT/s × 4 lanes × 128b/130b ÷ 8` |
| NVMe realistic (with cuFile, NVMe queues full) | **~6-7 GB/s** | gdsio empirical on similar setups |
| Unified memory (LPDDR5X, GB10) | **~273 GB/s** | NVIDIA DGX Spark published spec |
| Blackwell HW-DE Deflate throughput | **order of ~50-80 GB/s** | NVIDIA public disclosures for B200; specific number needs a primary-source check; SM120 unverified |

Two ratios fall out of that table:

- **Memory : storage ≈ 40 : 1.** No reasonable compression ratio closes that gap. On already-quantized NVFP4 weights, Deflate/Snappy typically achieves 1.05-1.25× (high-entropy data, quantization has already destroyed the redundancy). Even at a generous 1.5× ratio you get ~10.5 GB/s effective read, still 25× slower than RAM. The "treat SSD as swap" dream dies here for dense whole-model-in-disk serving.

- **HW-DE throughput : NVMe throughput ≈ 10 : 1.** This is the good ratio. It means the decompression step is *never the bottleneck* — the SSD is. So the question collapses cleanly: how much of a problem can you solve when your effective read bandwidth is ~7-10 GB/s?

#### What 7-10 GB/s actually buys you

Cold-start a 70 GB NVFP4 weights file:
- Uncompressed, mmap / cuFile: 70 / 7 = **10 s**
- Compressed Deflate 1.2×: 58 / 7 + ~1 s decomp overhead = **~9 s**
- That's a ~10% cold-start win, achievable without exotic tooling.

Cold-start a 200 GB model (dense, does not fit in unified memory):
- Moot. If it doesn't fit, loading it faster doesn't help.

Per-token streaming for dense decode (worst case: reload everything every token):
- Even a 12 GB layer at 7 GB/s is ~1.7 s. Decode budget is ~20-50 ms. Off by ~35-85×. Dead.

Per-layer streaming for MoE decode (active experts only):
- Qwen2-MoE-style, ~6-8 of 64 experts active per layer, ~128-256 MB of compressed weights per layer per step, prefetched one layer ahead. At 7 GB/s that's 18-37 ms per layer. **This lands in the decode budget.** See §3.1.

### 1.3 cuFile on unified memory — the semantic weirdness

Classic GPU Direct Storage (as marketed) solves this problem:

```
[NVMe] → DMA → [GPU VRAM]       # good: no CPU bounce
```

versus the naive path:

```
[NVMe] → [CPU RAM] → memcpy → [GPU VRAM]   # bad: CPU bounce
```

On GB10 unified memory, "GPU VRAM" and "CPU RAM" are physically the same LPDDR5X. The bounce buffer savings are zero by construction. So what is cuFile *actually* doing on Spark?

Three things, all still useful but less dramatic:

1. **Async queue depth.** cuFile's `cuFileReadAsync` lets you issue dozens of overlapping NVMe reads without thread-per-read scaffolding. Important because NVMe SSDs need queue depth ≥ 16 to saturate the link.
2. **Page-cache bypass (O_DIRECT path).** You get to not pollute the host page cache with 70 GB of weights you'll read once.
3. **Compat-mode fallback.** When HW/kernel support isn't present, cuFile falls back to a POSIX + cudaMemcpy path. On GB10 this fallback is *cheap* (unified memory), so even "broken" GDS setups degrade gracefully.

What you do *not* get on GB10:
- Meaningful CPU-offload (there's no bounce to offload)
- Datacenter-style RDMA-to-GPU magic (`libcufile_rdma` is installed but not relevant here)

**Net:** cuFile is a convenience library on Spark, not a speedup library. It's still the right abstraction to reach for, but don't expect the 2-3× read-bandwidth wins GDS advertises on H100 + NVMe rigs.

### 1.4 Compression ratios on NVFP4 — the reality check

NVFP4 weights are pre-packed into:
- **Mantissa data:** 4 bits per weight, near-uniform over the representable codes after quantization. Entropy is ~3.8-3.95 bits/symbol. Deflate/Snappy on this has almost nothing to compress.
- **Block scales:** FP8 e4m3 or FP32 per-block factors. These *do* have structure (locality within tensor). Typical 2-3× achievable.
- **Structural metadata:** shape, names, offsets. Trivially small, ignore.

Weighted overall ratio for a full NVFP4 checkpoint is likely **1.10-1.25×**. Claiming more without a measurement is fantasy. Before building anything, run:

```
# pseudocode (no implementation, just the experiment)
deflate(nvfp4_weights.bin) -> size_ratio
deflate(scale_factors_only.bin) -> size_ratio
snappy(nvfp4_weights.bin) -> size_ratio
```

If the ratio is <1.15× for the whole file, compression is not doing useful work — you'd be using the HW-DE as a "fast path from SSD to RAM that happens to also use a codec," which is fine but not the compelling story. If the ratio is >1.3× there may be codec + layout tuning worth doing (e.g., separating scales and mantissas into different streams).

**Tangent worth flagging:** there's real work in the "codecs tuned for quantized neural weights" space. E.g., Bitcomp (NVIDIA, lossless-int integer-friendly) or the ANS-based codecs that the LLM-quant community has been discussing. These aren't on the HW-DE list, so they'd run on the SMs, but might get 1.5-2× and still fit under the SM time budget.

---

## Part II — Prior art (the mac fans cracked it)

### 2.1 LLM in a Flash (Apple, Dec 2023)

Alizadeh et al., "LLM in a flash: Efficient Large Language Model Inference with Limited Memory" (arXiv 2312.11514, confidence high). This is almost certainly the work that generated the Twitter buzz. Core idea: serve an LLM from NVMe instead of DRAM by (a) using FFN sparsity to avoid loading most of each layer, (b) overlapping I/O with compute via windowing. Reports running models 2× larger than DRAM with substantial decode speedup vs naive loading; specific speedup multipliers vary by config and should be re-read in the paper before citing.

Relevance to us:
- **They don't use compression.** They rely on *sparsity* to reduce how much gets read. Their compression story is "the model is only 1 bit of overhead beyond what you would have read anyway," not "we decompress on the fly."
- **Their hardware** is M-series Apple Silicon: unified memory, narrower memory bus (~100-400 GB/s depending on SKU), NVMe at similar Gen4 speeds to Spark.
- **The Spark analogy is near-perfect.** If anything, GB10's higher memory bandwidth (~273 GB/s) and the HW-DE block give us *more* headroom than they had.

The nuance Apple's paper gets right that a naive "compress weights + stream them" approach misses: **the interesting variable is not bandwidth, it's how much you can skip reading entirely**. Compression reduces bandwidth cost per byte; sparsity reduces bytes read. On this hardware class, sparsity wins every time.

### 2.2 MLX and the "model splitting" Twitter buzz

The MLX community has been shipping lazy-load and "array splitting" primitives since late 2024. The specific thing getting buzz is probably [mlx-lm's sharded/lazy weight-loading](https://github.com/ml-explore/mlx-examples) — not a fundamentally new technique vs llama.cpp's mmap, but with better prefetch heuristics and first-class async. Worth reading their I/O layer if we're serious about this. The underlying mechanism is still memory-mapped weights leaning on macOS page-cache + unified memory, not compression.

### 2.3 llama.cpp `--mmap`

The OG version of the trick, going back to early 2023. Works because Linux/macOS mmap + a unified (or effectively-shared) memory system means the OS page-cache manager handles "which pages are hot" for you. Shocking how well this works in practice for dense models. Our baseline for any Spark weight-streaming project should be "does this beat `mmap` with a warm page cache?" If the answer is "no," we don't have a project.

### 2.4 FlexGen / DeepSpeed ZeRO-Inference

Both target the *dGPU + big-CPU-RAM + NVMe* tier hierarchy problem. They schedule weights across three levels with careful prefetch. The scheduling theory is good and directly portable to our two-level (unified / NVMe) problem. The actual code is heavy and x86-y; not a good port target. But the *papers* are good reading.

### 2.5 MoE-specific offloading (PowerInfer, DeepSpeed-MoE, Fiddler)

This is the most relevant prior art for us. PowerInfer (SJTU, 2023) and Fiddler (UW, 2024) both treat MoE experts as a streaming problem with hot/cold classification — exact arXiv IDs should be confirmed before citing in a paper. They don't compress, but their scheduling story (predict which experts are hot, prefetch the cold ones as needed) is exactly what you'd want to overlay on top of a HW-DE-accelerated stream.

### 2.6 Compressed-weights inference literature

General pattern from the compressed-inference literature: structured entropy codecs on FP16/BF16 weights land around 1.3-1.5×; on already-quantized (INT8/INT4/FP4) weights the ratio collapses toward 1.1-1.2×. This is the finding to verify before committing to anything — do a 30-minute ratio survey on one of our real NVFP4 checkpoints (§4, experiment 3). If I'm wrong and NVFP4 compresses well, the whole feasibility picture brightens.

---

## Part III — Where this could actually be fun on GB10

Three use cases, in declining order of how much I believe in them.

### 3.1 MoE expert streaming — the actual interesting one

**Concrete target landscape (verified against public specs, April 19 2026):**

| Model | Release | Total params | Active | NVFP4 weight size | Fits in 120 GB usable? |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 2026-04-16 | 35B | 3B (256 experts, 8+1 active) | ~17.5 GB | **Yes — trivially** |
| Mistral Small 4 | ~2026 | 119B | 6.5B | ~60 GB | **Yes** |
| DeepSeek V3 | 2024-12 | 671B | 37B | ~335 GB | **No — streaming required** |
| GLM-5 | 2026-02-11 | 744B | 40B (256 experts, top-8) | ~372 GB | **No — streaming required** |
| GLM-5.1 | 2026-03-27 (API) / 2026-04-07 (open) | 744B | 40B | ~372 GB | **No — streaming required** |
| Kimi K2 / K2.5 | 2026-01-27 | ~1.04T | 32B (384 experts) | ~520 GB | **No — streaming required** |
| Kimi K2.6 Code Preview | 2026-04-13 | ~1T | 32B | ~500 GB | **No — streaming required** |
| DeepSeek V4 | 2026-04 (preview/launch window) | ~1T | ~32-37B | ~500 GB | **No — streaming required** |

Two things jump out from the real table:

1. **Frontier labs have converged on ~1T total / ~32-40B active (3-5% sparsity).** DeepSeek V4, Kimi K2.5/K2.6, and GLM-5.x all land in that range. None of these fit in unified memory on GB10 at any quantization we're willing to ship. Streaming isn't optional for these — it's the *only* way they run on Spark.

2. **Qwen took the opposite route with 3.6-35B-A3B (3B active of 35B total, 256 experts).** Fits in <20 GB and leaves the whole box for KV cache. Excellent product positioning, doesn't help the streaming project. If bigger Qwen3.6 variants land later (397B Qwen3.5 sized at 3-5% active would still be ~200 GB FP4), they slot back into the streaming bucket.

3. **Active-weight fractions are falling fast.** Old-school Mixtral 8×22B ran at ~28% active; today's frontier is 3-5% active. That trend is a pure tailwind for expert streaming — the less weight activated per token, the narrower the bandwidth budget a streamer needs to hit.

**Setup for the math:** a MoE model with ~64 experts per layer, ~6-8 active per token, total weight size exceeding 120 GB so it cannot all fit in unified memory. Experts are the dominant term in total model size (>80% for typical MoE).

**What the math says:**

- Per-layer active-expert weight footprint: ~1 GB (FP4, 6-8 of 64 experts)
- Compressed: ~0.8 GB at 1.2× Deflate
- NVMe read at 7 GB/s: ~115 ms
- HW-DE decomp at 60 GB/s: ~13 ms
- Total stream-in time: ~128 ms per layer, *if you wait serially*

That's too long by itself. But:

- **Prefetch overlap.** You stream layer *L+1*'s experts while compute runs on layer *L*. Compute per layer at batch=1 is in the 20-50 ms range for a mid-size MoE on GB10, giving you a ~25-50% overlap budget against the stream.
- **Expert prediction.** PowerInfer/Fiddler-style predictors can pre-warm the right experts, reducing tail-latency fetches.
- **Cache of hot experts.** Keep 70% of experts (the frequent ones) resident in unified memory; stream only the cold 30%. This dramatically reduces the miss rate and is exactly what Apple does with FFN sparsity.

**What I believe:** with the HW-DE path active on SM120 and a sensible caching/prefetch policy, you could fit a ~200 GB MoE on a 128 GB box, at ~1.5-3× decode slowdown vs fully-resident. That's a genuinely new capability for this hardware tier. It's also the first place I'd prototype.

**Blocker / unknown:** whether the HW-DE is actually fast on SM120. If it's SM-only, decomp at 15-25 GB/s on the SMs is still plausible and the project still kind-of-works, but now you're burning compute time for decomp during decode, which eats into the overlap budget. Measure before building.

### 3.2 Multi-model hot-swap

**Setup:** keep a library of 2-5 compressed model checkpoints on NVMe. Swap between them in <3 s without a full process restart.

**What the math says:**

- A 60 GB NVFP4 checkpoint at 7 GB/s = ~8.5 s uncompressed, ~7-8 s compressed.
- That's "cold swap." For "warm swap" (evict current model, stream in next) you can overlap tear-down and bring-up.
- Realistically: 3-8 s model swap, vs ~30-90 s for full Python-process restart + Triton autotune + CUDA graph capture.

**Where this helps us:** we have three first-class use cases on Spark — serving (Qwen NVFP4), Hermes local agent, and kernel-debug / eval work — that each want a different model. Today you restart vLLM to switch. Hot-swap would make nvllm feel like a different product.

**Blocker / unknown:** CUDA graph capture state and Triton JIT caches would need to be per-model, and the current vLLM entrypoint assumes one model for process lifetime. This is a bigger refactor than the I/O code itself.

### 3.3 Cold-start acceleration

**Setup:** make `uv run vllm serve` produce a ready-to-accept-requests server in <10 s for a 70 GB model (currently ~30-45 s).

**What the math says:**

- Weight load is usually ~30-60% of cold start. The rest is autotune + graph capture + Python import.
- Compressing the weights file buys at most ~10-20% on the weight-load portion, or ~3-8 s of end-to-end cold-start reduction.

**Verdict:** nice-to-have, not a headline. If we do it at all it's a free ride-along from the MoE work.

### 3.4 Things I considered and don't believe in

- **Compressing FP16 / BF16 serving.** Higher ratios (~1.4×), but we already run NVFP4. Going backwards on quant just to stream faster is silly.
- **Dense model >128 GB streaming.** Bandwidth math is punitive without sparsity. Would need a fundamentally different (sparsity-exploiting) approach, and Apple already did it.
- **Using the HW-DE for KV cache compression.** Interesting-sounding, but KV cache reads/writes are the compute kernel's inner loop; adding an async decomp round-trip blows the roofline. Leave it alone.

---

## Part IV — Experiments to run before writing code

In rough order, cheapest first:

1. **Microbench nvCOMP Deflate on Spark.** Compress a 1 GB random buffer and a 1 GB NVFP4 weights slice. Run `nvcompBatchedDeflateDecompressAsync`. Record: compression ratio, decomp GB/s, peak SM utilization. Answers "is the HW-DE alive on SM120?" in one afternoon. (~1 hr)

2. **Microbench cuFile on Spark.** Use `gdsio` to measure actual NVMe read bandwidth with realistic queue depths, comparing against `mmap` + `madvise`. Answers "does cuFile actually help on unified memory?" (~30 min)

3. **Compression-ratio survey on NVFP4 checkpoint.** Per-tensor-class ratios (mantissas vs scales vs metadata) for one of our existing NVFP4 models. Tells us whether to bother separating streams, or whether a single Deflate pass is fine. (~1 hr)

4. **Overlap simulation.** No code — a spreadsheet. Model "what fraction of MoE layers can we overlap stream + compute for" on a Qwen-style MoE at Spark's compute/bandwidth numbers. This determines whether §3.1 is viable before we write a single line of it. (~1 hr)

Total cost to kill-or-validate the entire concept: ~3-4 hours of microbenching and math. No Docker rebuilds, no kernel work.

---

## Closing thought

The tempting-but-wrong framing is "compression gives us bigger effective memory." It doesn't — not at the ratios NVFP4 allows. The correct framing is "HW-DE + cuFile gives us a *fast lane* from NVMe to compute, and the right data structure to hang on that fast lane is **sparse activation** (MoE experts, FFN windowing à la LLM-in-a-Flash), not dense weights."

Two things have changed the stakes of this framing in the last six months:

1. **The model trend matches the hardware.** Frontier models went sparse-MoE with ever-smaller active fractions — exactly the shape that makes streaming feasible. A GB10 box with a working expert-streaming pipeline could plausibly run models that would need 300-500 GB of unified memory to hold dense. Nobody is shipping 500 GB unified SoCs for desktops.

2. **The pieces are all shipped.** cuFile + gds-tools are already installed on this box. nvCOMP 4.2+ exposes the Blackwell DE as a transparent backend. The only unshipped piece is a vLLM-compatible expert-streaming weight loader with a predictor and prefetch queue. That's a project, not a research program.

The MacFan comparison is apt and also flattering to us — Apple made "LLM in a flash" work using only sparsity and mmap, on hardware with narrower memory buses than Spark. We get sparsity *plus* the DE block *plus* more memory bandwidth. If this turns into a real effort, the target isn't "match the Mac trick" — it's "be the reference way to run frontier MoE on a 128 GB unified SoC."

§4's four microbenches are the next step. Do those before writing any loader code.

---

## Sources

**nvCOMP / cuFile:**
- [nvCOMP release notes (4.2.0 — Blackwell HW Decompress Engine)](https://docs.nvidia.com/cuda/nvcomp/release_notes)
- [nvCOMP C API docs](https://docs.nvidia.com/cuda/nvcomp/c_api)
- Local verification: `dpkg -l | grep cufile`, `/usr/local/cuda/gds/tools/gdsio --version`, `lsblk`, `nvidia-smi` (2026-04-19)

**Target model specs (as of 2026-04-19):**
- [Qwen3.6-35B-A3B on Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Kimi K2.5 — Moonshot AI](https://platform.moonshot.ai/) / [Kimi K2.5 GitHub](https://github.com/MoonshotAI/Kimi-K2.5)
- [Kimi K2.6 Code Preview blog](https://kimi-k2.org/blog/23-kimi-k2-6-code-preview)
- [DeepSeek V4 spec roundup (NxCode)](https://www.nxcode.io/resources/news/deepseek-v4-release-specs-benchmarks-2026)
- [GLM-5 specifications (NxCode)](https://www.nxcode.io/resources/news/glm-5-open-source-744b-model-complete-guide-2026)
- [GLM-5.1 deployment guide (Spheron)](https://www.spheron.network/blog/deploy-glm-5-1-gpu-cloud/)
- [2026 Frontier LLM architectures summary (Largo)](https://largo.dev/articles/frontier-llm-architectures-2026/)

**Prior art:**
- Alizadeh et al., "LLM in a Flash," [arXiv:2312.11514](https://arxiv.org/abs/2312.11514)
- llama.cpp mmap implementation (repo)
- MLX lazy loading (mlx-lm repo)
- PowerInfer / Fiddler — MoE offloading; arXiv IDs to verify before paper citation

