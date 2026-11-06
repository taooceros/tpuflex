# TPUFlex

TPUFlex is a research system for serving LLMs on heterogeneous TPU and GPU slices. It switches a running deployment between parallelism layouts (for example TP16 ↔ DP2×TP8) and scales capacity up and down, without paying a full cold start each time.

This repository is an **index**. The code lives in five forks of the upstream vLLM / llm-d projects. They are included here as git submodules, pinned to the research branches.

```bash
git clone --recursive --shallow-submodules https://github.com/taooceros/tpuflex.git
cd tpuflex
# Rebuild the single-tree research workspace from the five forks:
python llm-d/experimental/heterogeneous-serving/assemble.py --forks-root . --output ./workspace
```

## Repositories

| Submodule | Fork | Owns |
|---|---|---|
| [`llm-d`](https://github.com/taooceros/llm-d) | llm-d | Orchestration, layout-switch controller, CLI, campaign drivers, analysis scripts, docs and plans (`experimental/heterogeneous-serving/`) |
| [`tpu-inference`](https://github.com/taooceros/tpu-inference) | vLLM TPU backend (vLLM-JAX) | TPU worker and device runtime: prespawned workers, executable bank, weight snapshots, restore graph, ICI transfer, unified runtime |
| [`vllm`](https://github.com/taooceros/vllm) | vLLM | Engine glue: engine actor, in-process engine, KV transport, scheduler, HTTP serving |
| [`llm-d-router`](https://github.com/taooceros/llm-d-router) | llm-d router | Gateway / request routing under layout changes |
| [`llm-d-async`](https://github.com/taooceros/llm-d-async) | llm-d async | Workloads, experiment and benchmark harnesses, cost and policy models, autoscaling simulators |

Each fork's `experimental/heterogeneous-serving/migration.json` lists the files that fork owns. `assemble.py --check` verifies that the five manifests agree.

## Branch stack (same in every fork)

```
research/heterogeneous-tpu-20260918      base snapshot
└─ pr/hetero-local-20261012              local research commits
   └─ pr/hetero-chain-20261012           main research line, publication 1
      ├─ side branches (2026-10-12)      first-switch, load-profiler, pathways-reshard,
      │                                  ici-benchmark, in-place-engine, unified-runtime, reconfig-levers
      └─ pr/hetero-chain-20261105        main research line, publication 2  ← submodules pin this
         └─ side branches (2026-11-05)   see below
```

`llm-d` has one more layer: `pr/hetero-local-20261105` sits between `pr/hetero-chain-20261012` and `pr/hetero-chain-20261105`. `llm-d-router` has no `pr/hetero-chain-20261105`, because the new main line adds no router files, so its submodule pins `pr/hetero-chain-20261012`.

Every branch is an open pull request in its fork; the PR list is the easiest way to read the diffs: [llm-d](https://github.com/taooceros/llm-d/pulls) · [tpu-inference](https://github.com/taooceros/tpu-inference/pulls) · [vllm](https://github.com/taooceros/vllm/pulls) · [llm-d-router](https://github.com/taooceros/llm-d-router/pulls) · [llm-d-async](https://github.com/taooceros/llm-d-async/pulls).

### Side branches (`pr/hetero-<name>-20261105`)

| Name | Topic | Forks |
|---|---|---|
| `matrix-tpu-jax` | Cold-start matrix: TPU × vLLM-JAX | llm-d |
| `matrix-tpu-torch` | Cold-start matrix: TPU × vLLM-torch (includes torch-TPU rewind / load) | llm-d, vllm, llm-d-async |
| `matrix-gpu-jax` | Cold-start matrix: H100 × vLLM-JAX, leave-one-out ablation, GPU autoscaling | llm-d, llm-d-async |
| `matrix-gpu-torch` | Cold-start matrix: H100 × vLLM-torch, serverless baseline, GPU autoscaling | llm-d, llm-d-async |
| `autoscale` | Burst autoscaling model and controller | llm-d, llm-d-async |
| `autoscale-cost` | Autoscaling cost simulator; 2×2 matrix report | llm-d, llm-d-async |
| `in-engine-dp-switch` | TP16 → DP2×TP8 switch inside one engine; virtual-process block runtime; full-mesh controller | llm-d, vllm, tpu-inference, llm-d-router |
| `pre-tpu-rewind` | Rewind to pre-TPU state; portable executable bank | llm-d, vllm, tpu-inference, llm-d-async |
| `jax-0112-image` | Container image build for JAX 0.11.2 | llm-d, vllm, tpu-inference |
| `helper-overlap`, `helper-snapshot-route`, `helper-results-analysis` | Analysis helpers | llm-d |

The main line for 2026-11-05 also includes the portable executable bank, the autoscaling ablation, the TPU end-to-end matrix cell and the llm-d autoscaling comparison.

To check out one side branch in every fork that has it:

```bash
git submodule foreach 'git fetch --depth 1 origin pr/hetero-in-engine-dp-switch-20261105 && git checkout FETCH_HEAD || true'
```

## Evidence

Raw experiment results are not stored in the forks. They are published as release assets on the `llm-d` fork, with a sha256 asset index:

- [research-evidence-20261105](https://github.com/taooceros/llm-d/releases/tag/research-evidence-20261105)
- [research-evidence-20261012](https://github.com/taooceros/llm-d/releases/tag/research-evidence-20261012)
- [research-evidence-20260918](https://github.com/taooceros/llm-d/releases/tag/research-evidence-20260918)

`llm-d/experimental/heterogeneous-serving/research/restore_evidence.py` downloads the assets, verifies them and unpacks them into `results/`.

## Updating the pins

```bash
git submodule update --remote   # follows the branch recorded in .gitmodules
git commit -am "Bump submodule pins"
```
