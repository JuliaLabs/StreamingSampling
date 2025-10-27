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
    println("Checking sample size.")
    sme = StreamMaxEnt(file_paths; chunksize=1000, subchunksize=100)
    inds = StreamingSampling.sample(sme, 100)
    @test length(inds) == 100
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

