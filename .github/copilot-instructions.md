## Purpose
Help an AI coding agent become productive quickly in this Julia project (StreamingSampling).

## Big picture (what to read first)
- Top-level entry: `src/StreamingSampling.jl` — it pulls together the project by including the key modules (features, chunk iterator, weights, UPmaxentropy, sampling).
- Sampling flow: `src/ApproxWeights.jl` -> `src/IncluProbsRelFreqs.jl` -> `src/UPmaxentropy/UPmaxentropy.jl` -> `src/Sampling.jl`.
- Data handling: streaming/chunking is implemented via `src/LazyChunkIterator.jl` and `src/ApproxWeights.jl` (see the channel-based `compute_weights` overloads).

## Key files to reference
- `src/StreamingSampling.jl` — project composition and external package list.
- `src/ApproxWeights.jl` — streaming-weight algorithm, merging/imputation (alpha/beta polynomials), and global weights layout.
- `src/Features.jl` — how raw elements are parsed into numeric feature vectors (`create_feature`, `create_features`).
- `src/Sampling.jl` — sampling entrypoints; `sample(sampler, n)` calls `inclusion_prob` and `UPmaxentropy`.
- `src/UPmaxentropy/UPmaxentropy.jl` — core deterministic selection used by `sample`.
- `examples/atomistic/cluster/*.sh` — runnable example scripts showing expected invocation patterns for experiments.

## Important patterns and conventions
- Data format: elements are whitespace-separated numeric vectors (see `create_feature` in `src/Features.jl`).
- Chunking: functions accept either a vector of file paths, a vector of in-memory items, or a `Channel` produced by `chunk_iterator` (see `compute_weights` overloads in `src/ApproxWeights.jl`).
- compute_weights behavior: it 1) initializes with a warm-up chunk, 2) iteratively replaces slots in a fixed-size buffer, computes features, then recomputes weights and merges them into a global weights vector (`gws`). Pay attention to `chunksize`, `subchunksize`, `buffersize`, `max`, and `randomized` parameters.
- Concurrency: `create_features` uses `Threads.@threads`; other functions assume in-place updates of arrays (mutable reuse is common). Avoid introducing heavy copies unless necessary.

## Developer workflows (how to run / debug)
- Julia environment: prefer `julia --project=.` so the package environment can be defined if/when a `Project.toml` is added.
- Quick interactive load (assumes required packages installed):

```bash
# start a one-shot Julia run that loads the project files
julia --project=. -e 'include("src/StreamingSampling.jl"); println("Loaded StreamingSampling")'
```

- To run example experiment scripts (shell wrappers live in examples): run the scripts in `examples/atomistic/cluster/` (they call Julia with project-aware commands).
- To test small pieces interactively, open the REPL and `include` modules, then call functions like `generate_data` or `compute_weights`.

Example (assumptions: you have a concrete sampler subtype available in examples or your REPL):

```julia
include("src/StreamingSampling.jl")
file_paths = ["/tmp/data1.txt", "/tmp/data2.txt"]
generate_data(file_paths; N=1000, feature_size=50)
# sampler must be a concrete subtype defined elsewhere in examples
# gws = compute_weights(my_sampler, file_paths; chunksize=1000, subchunksize=100)
```

## Integration points & external deps
- Top of `src/StreamingSampling.jl` lists required packages: Determinantal, Distances, Distributed, Distributions, Optim, Plots, Random, StatsBase, Roots, etc. Ensure the runtime environment has these installed.
- UPmaxentropy is used as the deterministic step and lives under `src/UPmaxentropy/`.

## What an AI agent should do first
1. Load `src/StreamingSampling.jl` in a REPL to validate include paths and missing packages.
2. Trace `compute_weights` in `src/ApproxWeights.jl` to understand data movement and mutation (it's central to streaming logic).
3. Inspect `create_features` in `src/Features.jl` to confirm parsing rules when changing I/O or feature dimensions.

## Common pitfalls to avoid
- Don't assume immutability: many arrays are mutated in-place and reused across iterations.
- Respect chunk/subchunk sizes: changing `subchunksize` without adjusting `chunksize` or call sites can break initialization loops.
- When adding multithreading, mirror `create_features`'s use of `Threads.@threads` and be mindful of thread-safety for shared arrays.

If anything here is unclear or you'd like me to expand examples (concrete sampler types, end-to-end run commands), tell me which parts to expand.
