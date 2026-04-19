# llama.cpp — Strix Halo fork

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with experimental changes targeting **AMD Strix Halo** (`gfx1151`, RDNA 3.5 iGPU + Zen 5 + 128 GB LPDDR5x unified memory), found in the Framework Desktop and similar systems.

The goal is a reproducible, benchmarked set of patches that meaningfully improve inference performance on this specific chip — primarily for agentic coding workloads, where long-context prompt processing and TTFT dominate the UX.

## Findings

| # | Finding | Impact | Status |
|--:|---|---|---|
| 1 | [Quantized KV cache collapses throughput at depth](kv-cache.md) | **17× pp @ d=16k** on Qwen 3.6; V-quant is the dominant cost | Config fix only; no patch needed |
| 2 | [FA dispatcher gates RDNA3.5 out of MMA_F16 kernel](fa-dispatcher.md) | Attempted 1-line patch; **abandoned** | See doc — blocked on MMA device code not compiled for gfx1151 |
| 3 | [UMA / `integrated = false`](uma-integrated.md) | Originally flagged as likely biggest win; research says otherwise | **Researched, deprioritized** — narrow on HIP APUs |
| 4 | [ROCm config flags: unroll-threshold + `HIPBLASLT_BATCHED=0`](rocm-config.md) | Community reports 2× pp on other models; null on Qwen 3.6 | **Bench null, kept on** as AMD-recommended safety nets |
| 5 | [MMQ tile/nwarp tuning for gfx1151 (port of PR #21344)](mmq-rdna3_5.md) | **+27% pp @ d=0, +17% pp @ d=16k** on Qwen 3.6 Q4_K_XL; tg128 flat | **Kept** on master (commit `d8ad713`) |
| 6 | [rocWMMA FA tuning for gfx1151 (port of PR #16827)](rocwmma-tuned.md) | **Flat** on Qwen 3.6 (D=256 dodges 3 of 4 D-gated knobs); lhl measured +35-65% pp @ depth on D≤128 models | **Kept** on master (commit `1be00ab8`) — zero-cost on current workload, real wins on any D≤128 model swapped in |

See [NOTES.md](NOTES.md) for the initial survey of other possible optimization sites in the llama.cpp source (most not yet pursued).

## Benchmarks

- [Qwen 3.6 35B-A3B baseline on gfx1151](qwen3.6-baseline.md) — establishes production-config numbers and the f16-KV recovery.

## Approach: one branch, one commit at a time

Single working branch: `master` on the fork (this repo). It tracks `ggml-org/llama.cpp` upstream plus this `strix-halo/` docs folder plus any accumulated, validated patches.

Rather than maintain a separate branch per patch + an integration branch, each optimization attempt is a single commit landed on master. The workflow per attempt:

1. **Write up the hypothesis** in a new markdown doc under `strix-halo/` before touching code. Link the source lines you plan to change, state what you expect to measure.
2. **Make the change** as one commit on master.
3. **Build & benchmark** via the server-configs Dockerfile (see [the full loop below](#per-attempt-loop-code--lab--bench)). Always pin to a SHA, never a branch name — Docker caches the git-clone layer by command string.
4. **Decide**: if the bench shows a real gain (outside noise, across the matrix of depths/quants we care about), the commit stays. If it's null or negative, revert the commit and leave the doc as a record of the dead end, with the reason annotated at the bottom.

This keeps the history as a log of "tried this, here's what happened" rather than a pile of speculative branches. Docs accumulate even when patches don't — we've already gotten two concrete findings (KV quant, FA dispatcher dead-end) out of exactly this rhythm.

### Per-attempt loop: code → lab → bench

The lab host (`lab.28r.net`) is provisioned by [server-configs](https://github.com/justinappler/server-configs) Ansible. The llamacpp container is built from this fork's SHA pinned in `ansible/inventory/group_vars/lab/vars.yaml`. Full round-trip:

**1. Commit + push to the fork.**

```bash
# In this repo:
git add -A
git commit -m "…"
git push strix-halo master            # strix-halo = justinappler/llama.cpp-strix-halo

# Grab the full SHA (Docker's git-clone cache is keyed by the full command string)
git rev-parse HEAD
```

**2. Bump the pinned SHA in server-configs.**

Edit [ansible/inventory/group_vars/lab/vars.yaml](https://github.com/justinappler/server-configs/blob/main/ansible/inventory/group_vars/lab/vars.yaml):

```yaml
llamacpp_version: <the full 40-char SHA from step 1>
```

Commit and push that change (same branch name does not invalidate the Docker cache — only a different SHA does, which is why we pin by SHA and not branch).

**3. Apply with Ansible.** From the `server-configs` repo root:

```bash
make deploy-lab           # lab.28r.net on local network
# or
make deploy-lab-remote    # via home.28r.net:61388
```

Under the hood this runs `ansible-playbook playbooks/lab.yaml --limit lab[-remote]`. The playbook rebuilds the `llamacpp-server:local` image with the new `LLAMACPP_VERSION`, restarts the container, and waits for the API to come back. Expect **a few minutes** (ccache is mounted, so incremental rebuilds are fast; cold ROCm tarball + full source build can take up to the 2400s `service_timeout`).

**This is a shared step — wait for the playbook to complete before benching.**

**4. Bench on the lab.** SSH to the host, then run `llama-bench` inside the container. The `llamacpp` container runs the server on deploy but does **not** auto-load a model, so it won't contend for GPU memory — no need to stop/start it around the bench. Two things to watch for on the `docker run`:

- The image's `ENTRYPOINT` is `/app/entrypoint.sh` (which launches the server). To run a different binary, override with `--entrypoint=/app/llama-bench` rather than appending the bench path as a positional (that gets fed to the entrypoint as an arg and fails with `error: invalid argument: /app/llama-bench`).
- Bench args come after the image name as plain positionals — no `--` separator needed once `--entrypoint` is set.

```bash
ssh lab

# Pick model + flags to match strix-halo/qwen3.6-baseline.md
docker run --rm --entrypoint=/app/llama-bench \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc host \
  -v /srv/models:/models:ro \
  llamacpp-server:local \
    -m /models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -b 4096 -ub 2048 -ngl 999 -mmp 0 -p 512 -n 128 -r 3 \
    -d 0,2048,8192,16384
```

Models are pre-pulled into `/srv/models/<hf-repo>/`; see [models.ini](https://github.com/justinappler/server-configs/blob/main/services/llamacpp/files/models.ini) for the canonical paths. Cross-check the bench flags against [qwen3.6-baseline.md](qwen3.6-baseline.md) so numbers stay comparable — `-b 4096 -ub 2048 -ngl 999 -mmp 0` is the agreed baseline shape.

**5. Record + decide.** Paste the `llama-bench` table into the attempt's `strix-halo/<topic>.md`. Keep the commit if the bench shows a real gain across the depth/quant matrix; revert and annotate if null/negative.

### If you just want to try an upstream SHA (no fork commit)

Short-circuit: point `llamacpp_version` at an upstream `ggml-org/llama.cpp` SHA and also override the repo URL in the Ansible env (`LLAMACPP_REPO=https://github.com/ggml-org/llama.cpp.git`). Useful for bisecting upstream regressions without carrying a fork commit. Revert both when done.

## Keeping up with upstream

Upstream `ggml-org/llama.cpp` moves daily. Resync cadence is roughly every few days.

```bash
# Fetch upstream (origin = ggml-org/llama.cpp)
git fetch origin

# Rebase fork master onto upstream. Docs-only changes shouldn't conflict;
# if a patch commit conflicts with an upstream change that supersedes it,
# drop the commit during rebase and annotate the doc accordingly.
git checkout master
git rebase origin/master
git push -f strix-halo master          # strix-halo = justinappler/llama.cpp-strix-halo
```

## Build

See [server-configs Dockerfile](https://github.com/justinappler/server-configs) for a working multi-stage build against TheRock ROCm nightly that the upstream llama.cpp Dockerfile does not yet support on gfx1151. Official ROCm apt packages ship broken `gfx1151` `.hsaco` kernels — see [ROCm/ROCm#6042](https://github.com/ROCm/ROCm/issues/6042).

Point `LLAMACPP_VERSION` at a specific commit SHA on this fork's master (or on upstream) to rebuild. Branch names work but Docker caches the clone layer by command string — always use SHA when iterating.
