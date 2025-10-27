module StreamingSampling

using Base.Threads
using Determinantal
using Distances
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics
using StatsBase

# Sampler abstract type
abstract type Sampler end

# Chunk iterator
include("LazyChunkIterator.jl")

# Features
include("Features.jl")

# Approximated weights
include("Weights.jl")

# Inclusion probabilities
include("IncluProbs.jl")

# Relative frequencies
include("RelFreqs.jl")

# Sampling
include("UPmaxentropy.jl")
include("Sampling.jl")
include("StreamMaxEnt.jl")


export Sampler, StreamMaxEnt, compute_weights, sample, chunk_iterator

end # module

