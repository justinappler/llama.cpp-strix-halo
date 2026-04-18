# UMA / `integrated = false` — research brief (next session starting point)

**Status: research only, no patch yet.** This doc is the hypothesis writeup per the single-commit workflow ([README.md](README.md#approach-one-branch-one-commit-at-a-time)). The next working session should finish the research here before touching code.

## Why this is the target

From [NOTES.md #1](NOTES.md#1-zero-copy-for-integrated-gpus-is-globally-disabled) — the author's own pick as "likely the single biggest win." The HIP backend hardcodes `integrated = false` at [ggml/src/ggml-cuda/ggml-cuda.cu:243](../ggml/src/ggml-cuda/ggml-cuda.cu#L243), with a comment referencing upstream issue `#15034`. That flag gates a UMA fast-path (kernels operating directly on host/pinned buffers instead of requiring a DeviceLocal copy) at [ggml-cuda.cu:4085](../ggml/src/ggml-cuda/ggml-cuda.cu#L4085) and [ggml-cuda.cu:5125](../ggml/src/ggml-cuda/ggml-cuda.cu#L5125).

On Strix Halo, "VRAM" is a GART carve-out of the same LPDDR5X the CPU uses, so the forced copy is pure waste of bandwidth and memory. The linked issue likely revealed a real bug, but the blanket `false` is almost certainly overbroad for gfx1151 specifically.

## Context the agent should load first

Read these in this order — they're short:

1. [README.md](README.md) — single-branch workflow, approach
2. [NOTES.md](NOTES.md) — full survey, item #1 is this
3. [kv-cache.md](kv-cache.md) — the one validated finding so far (TILE + quant KV collapses at depth; prod uses f16/f16)
4. [fa-dispatcher.md](fa-dispatcher.md) — a dead end worth understanding before the next attempt; the pattern there (dispatcher looks broken but device code is the real blocker) is the shape of trap to watch for

## Research tasks

No code changes yet. Deliverable is an expansion of this doc with findings from the following:

1. **Read upstream issue [ggml-org/llama.cpp#15034](https://github.com/ggml-org/llama.cpp/issues/15034)**. What specifically broke when `integrated = true`? Which hardware was affected? Is there a linked PR that disabled the flag, and does its diff include anything besides the hardcode? Does the issue describe a bug that's since been fixed elsewhere in ggml-cuda?

2. **Read the flag's consumers in-tree**:
   - [ggml-cuda.cu:243](../ggml/src/ggml-cuda/ggml-cuda.cu#L243) — the hardcode itself, with full surrounding context
   - [ggml-cuda.cu:4085](../ggml/src/ggml-cuda/ggml-cuda.cu#L4085) — first consumer (likely buffer allocation path)
   - [ggml-cuda.cu:5125](../ggml/src/ggml-cuda/ggml-cuda.cu#L5125) — second consumer (likely kernel launch / memory access)

   For each site, figure out: what does `integrated=true` actually change? Is the answer "skip a cudaMemcpy," "change a buffer allocator," "allow a different kernel variant," or all of the above? Capture the answer in plain language in a new section of this doc.

3. **Check for prior attempts.** Search the llama.cpp git log for commits touching `integrated` in ggml-cuda. Is there anyone who tried to re-enable it conditionally? What arch/configuration did they gate on, and why was it reverted (if it was)?

4. **Check the HIP/ROCm side specifically.** The issue is in `ggml-cuda.cu` but the code is compiled as HIP for AMD. Does the HIP runtime expose UMA detection (`hipDeviceGetAttribute(hipDeviceAttributeIntegrated)` or similar)? What does it return on gfx1151? If it already returns `1`, then the hardcoded `false` is literally overriding correct runtime detection and the narrower fix is obvious.

5. **Frame the experiment.** Based on the above, propose the narrowest possible change that re-enables the UMA path for gfx1151 while preserving whatever guard `#15034` required. Examples of "narrow":
   - Gate on `hipDeviceGetAttribute(integrated) == 1`
   - Gate on a specific CC (RDNA3.5 only)
   - Gate on an env var for A/B testing
   Pick the one that most directly tests the hypothesis without breaking anything else. Write the proposed change as a diff block in this doc, but **do not commit it yet** — surface it to the user first.

## Known environmental state (don't re-derive)

- **Lab host:** Framework Desktop, gfx1151, Ubuntu 24.04, 128 GB LPDDR5X, container deployed via ansible playbook in `~/Projects/server-configs` (lab inventory, group `lab`). User runs the playbook — do not edit `.env` or anything on the lab host directly.
- **Current production commit:** `309b410e2` on fork master (upstream + docs, no code patches). Baseline performance is documented in [qwen3.6-baseline.md](qwen3.6-baseline.md).
- **Build cost:** cold ~80s with warm ccache; SHA change → full recompile of changed TUs. Docker caches git-clone by command string, so **pin SHA, not branch** in `ansible/inventory/group_vars/lab/vars.yaml` when testing.
- **Bench invocation** (headline 4-run f16/f16 matrix, ~5 min):

  ```bash
  ssh lab 'docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc host --ulimit memlock=-1:-1 \
    -v /srv/models:/models:ro \
    -e ROCBLAS_USE_HIPBLASLT=1 -e HSA_ENABLE_SDMA=0 \
    -e GPU_MAX_HEAP_SIZE=100 -e GPU_MAX_ALLOC_PERCENT=100 \
    --entrypoint /app/llama-bench \
    llamacpp-server:local \
    -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -p 512 -n 128 -r 3 -b 4096 -ub 2048 -ngl 999 -mmp 0 -fa 1 \
    -ctk f16 -ctv f16 -d 0,16384'
  ```

## Watch out for

- **Runtime-vs-device-code trap.** The FA dispatcher dead end in [fa-dispatcher.md](fa-dispatcher.md) looked like a one-line dispatcher fix and turned out to need cmake changes because the kernel had no device code for gfx1151. For UMA, the equivalent question is: is the fix *really* just flipping the flag, or does the UMA path depend on buffer-type flags, driver features, or kernel variants that don't exist on the HIP side? Check the consumers at lines 4085 and 5125 before assuming a one-line patch works.
- **Don't touch these.** KV cache config (prod is on f16/f16 for a reason — [kv-cache.md](kv-cache.md)). FA dispatcher (dead end). `strix-halo/fa-mma-rdna35` branch on the remote (user may delete it — don't commit new work there).
- **The user runs the playbook, not you.** When a test requires a rebuild, bump `llamacpp_version` in `ansible/inventory/group_vars/lab/vars.yaml` and hand it off. Don't ssh to the lab and run things by hand.
