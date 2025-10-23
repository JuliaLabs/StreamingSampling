module StreamingSampling

"""StreamingSampling

Top-level module for the StreamingSampling package. This module re-exports
the key types and functions from the `src/` files so examples can `using StreamingSampling`.
"""

using Base.Threads
using Determinantal
using Distances
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics
using StatsBase

# Samplers
abstract type Sampler end

# Chunk iterator
include("LazyChunkIterator.jl")

# Features
include("Features.jl")

# Approximated weights
include("ApproxWeights.jl")

# Inclusion probabilities and relative frequencies
include("IncluProbsRelFreqs.jl")

# Sampling
include("UPmaxentropy.jl")
include("StreamMaxEnt.jl")
include("Sampling.jl")


export Sampler, StreamMaxEnt, compute_weights, sample

end # module

