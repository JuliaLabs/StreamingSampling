using StreamingSampling
using Test
using Random
Random.seed!(123)
using LinearAlgebra
using StatsBase
using Statistics

@testset "StreamingSampling.jl" begin
    file_paths = ["data/data1.txt",
                  "data/data2.txt",
                  "data/data3.txt",
                  "data/data4.txt"]
    sme = StreamMaxEnt(file_paths; chunksize=1000, subchunksize=100)
    n = 2351
    inds = StreamingSampling.sample(sme, n)
    ps = StreamingSampling.inclusion_prob(sme, n)
    
    println("Checking sample size.")
    @test round(Int, sum(ps)) ≈ length(inds)
    
    println("Checking sum(ps) ≈ n.")
    @test round(Int, sum(ps)) ≈ n
    
    println("Checking 0<=ps_i<=1.")
    @test all(0 .<= ps .<= 1)
end

@testset "UPmaxentropy." begin
    ps=[0.07,0.17,0.41,0.61,0.83,0.91]
    inds = StreamingSampling.sample(ps)

    println("Checking sample size.")
    @test length(inds) ≈ round(Int, sum(ps))

    println("Checking second-order inclusion probabilities.")
    ps2 = StreamingSampling.UPmaxentropypi2(ps)
    @test ps ≈ diag(ps2)

    println("Checking relative frequencies.")
    reps = [StreamingSampling.UPmaxentropy(ps) for _ in 1:100_000];
    @test ps ≈ mean(reps) atol=1e-02
end

