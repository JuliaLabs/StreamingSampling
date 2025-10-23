# Large Scale Sampling

using Base.Threads
using Determinantal
using Distances
using Distributed 
using Distributions
using LinearAlgebra
using Optim
using Plots
using Printf
using Random
using Statistics
using StatsBase
using Roots

# Samplers
abstract type Sampler end
include("StreamMaxEnt.jl")

# Data generation
include("GenData.jl")

# Chunk iterator
include("LazyChunkIterator.jl")

# Features
include("Features.jl")

# Approximated weights
include("ApproxWeights.jl")

# Inclusion probabilities and relative frequencies
include("IncluProbsRelFreqs.jl")

# Sampling
include("UPmaxentropy/UPmaxentropy.jl")
include("Sampling.jl")

