# Example usage
using Random
Random.seed!(123)
using LinearAlgebra
using StatsBase,Statistics
using Plots

include("UPmaxentropy.jl")

pik=[0.07,0.17,0.41,0.61,0.83,0.91]
s = UPmaxentropy(pik)
println("Sample: ", s)
println("Sample size: ", sum(s), " (expected: ", round(Int, sum(pik)), ")")

# Test 1: Verify with diagonal of second-order probabilities
pi2 = UPmaxentropypi2(pik)
println("Second-order inclusion probabilities computed")
plot(pik,pik)
scatter!(pik,diag(pi2))

# Test 2: Verify with frequancies
reps = [UPmaxentropy(pik) for _ in 1:1000];
plot(pik,pik)
scatter!(pik,mean(reps))
