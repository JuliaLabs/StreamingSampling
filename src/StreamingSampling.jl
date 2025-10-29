module StreamingSampling

using Base.Threads
using Determinantal
using Distances
using Ipopt
using JuMP
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
#include("StreamWeights.jl")
include("StreamMaxEnt.jl")
include("Sampling.jl")


export Sampler, StreamMaxEnt, compute_weights, sample, chunk_iterator,
       inclusion_prob, UPmaxentropy

end # module

