# Agent Instructions for nvllm

## nvllm Fork Context

This is a fork of vLLM optimized for local inference on NVIDIA GB10 (DGX Spark).

- **Target:** single-user or small-group local serving, not datacenter deployment
- **Architecture:** SM120/SM121, 128 GB unified memory
- **All development and testing happens on-device** — no CI cluster
- **Custom patches live alongside upstream;** check upstream issues before debugging serving bugs
- **Readme is a more up to date representation of the repo state than this file.** Standards live here

---

> The upstream contribution policy below still applies if pushing patches back to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# If you are only making Python changes:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# If you are also making C/C++ changes:
uv pip install -e . --torch-backend=auto
```

### Running tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies.
# requirements/test.txt is pinned to x86_64; on other platforms, use the
# unpinned source file instead:
uv pip install -r requirements/test.in    # resolves for current platform
# Or on x86_64:
uv pip install -r requirements/test.txt

# Run a specific test file (use .venv/bin/python directly;
# `source activate` does not persist in non-interactive shells):
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

---

## 3. Kernel Work Attribution & Documentation

When adding, adapting, or deriving kernel code from external sources (other projects,
papers, reference implementations), follow these rules for traceability and proper credit.

### Pin the source

All references must link to a **specific commit hash**, not a branch or tag that can move.

```
# Good — pinned permalink
https://github.com/user/repo/blob/abc123def.../path/to/file.py#L42-L80

# Bad — moves over time
https://github.com/user/repo/blob/main/path/to/file.py
```

### Create an insights doc

For each external source, create a dedicated document in `docs/kernel-insights/`:

```
docs/kernel-insights/YYYY-MM-DD-<source-name>.md
```

Each doc must contain:
- **Source**: repository URL, pinned commit hash, license
- **What was borrowed**: list each piece (function, algorithm, approach, design pattern)
- **Why**: what problem it solves, why we chose this approach over alternatives
- **How it was adapted**: what changed for our target (SM121, vLLM integration, etc.)
- **Per-piece links**: every borrowed piece gets a direct permalink to the source line(s)

### README acknowledgment

Add an entry in the README's acknowledgments section with:
- Maintainer/author credit
- One-line description of what their work enabled
- Link to the insights doc for full details

### Verification checklist

Before committing kernel work derived from external sources:
1. All permalink URLs resolve (test them)
2. License compatibility confirmed
3. Insights doc is complete (no TBD/TODO)
4. README acknowledgment added
5. Commit message references the insights doc

## 4. Performance Evidence Standard

**Every performance claim must be backed by a committed nsys trace.** No rounding, no
cherry-picking — report what nsys reports.

### Trace directory structure

```
benchmarks/nvllm/traces/<area>/YYYY-MM-DD-<description>/
  baseline.nsys-rep        # before (or comparison backend)
  changed.nsys-rep         # after (or new backend)
  summary.md               # human-readable comparison
```

### summary.md requirements

Each summary must include:
- **Commit hash** of the code that produced the trace
- **Model and config** used (model name, kv-cache-dtype, max-model-len, etc.)
- **Kernel duration table** with exact μs values from nsys (no rounding)
- **How to reproduce** — exact commands to regenerate the trace

### Citing traces in docs and changelogs

```markdown
Decode attention improved 31% (142.3 μs → 98.1 μs).
([trace](benchmarks/nvllm/traces/cute_paged_attn/2026-05-15-initial/summary.md), commit abc123)
```

### Failure evidence

When baseline validation fails or a regression is found, also commit the trace under
`benchmarks/nvllm/traces/baseline_failures/YYYY-MM-DD/` with `docker_logs.txt`, the raw
API response, and a summary of what went wrong.

### nsys requirements

- Container must run with `--privileged` for CUPTI injection
- Use `--trace=cuda,nvtx` at minimum
- `.nsys-rep` files are binary but typically 5-20 MB — commit them alongside the summary

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.
- **Adding kernels, graph-captured ops, or numerics-aware code**:
  [`docs/contributing/design-tenets.md`](docs/contributing/design-tenets.md)
  — Invariants and receipts for graph replay, kernel contracts, and quant correctness.
