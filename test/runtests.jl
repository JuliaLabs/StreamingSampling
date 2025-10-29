using StreamingSampling
using Test
using Random
Random.seed!(123)
using LinearAlgebra
using StatsBase
using Statistics

@testset "Streaming weighted sampling" begin
    file_paths = ["data/data1.txt",
                  "data/data2.txt",
                  "data/data3.txt",
                  "data/data4.txt"]
    # Compute streaming weights
    ws = compute_weights(file_paths;
                         chunksize=200,
                         subchunksize=50)

    # Define sample size
    n = 1781

    # Sample by weighted sampling
    inds = StatsBase.sample(1:length(ws),
                            Weights(ws),
                            n;
                            replace=false)
    
    # Checks
    println("Checking sample size.")
    @test n == length(inds)

end

@testset "Streaming maximum entropy sampling" begin
    file_paths = ["data/data1.txt",
                  "data/data2.txt",
                  "data/data3.txt",
                  "data/data4.txt"]
    # Compute streaming weights
    ws = compute_weights(file_paths;
                         chunksize=500,
                         subchunksize=100,
                         normalize=false)

    # Define sample size
    n = 2351

    # Sample by UPmaxentropy
    ps = inclusion_prob(ws, n)
    s = UPmaxentropy(ps)
    inds = findall(s .== 1)
    
    # Check
    println("Checking sample size.")
    @test round(Int, sum(ps)) == length(inds)
    
    println("Checking sum(ps) ≈ n.")
    @test round(Int, sum(ps)) == n
    
    println("Checking 0<=ps_i<=1.")
    @test all(0 .<= ps .<= 1)
end

@testset "Maximum entropy sampling" begin
    # Sample by UPmaxentropy
    ps=[0.07,0.17,0.41,0.61,0.83,0.91]
    s = UPmaxentropy(ps)
    inds = findall(s .== 1)

    # Check
    println("Checking sample size.")
    @test length(inds) == round(Int, sum(ps))

    println("Checking second-order inclusion probabilities.")
    ps2 = StreamingSampling.UPmaxentropypi2(ps)
    @test ps == diag(ps2)

    println("Checking relative frequencies.")
    reps = [StreamingSampling.UPmaxentropy(ps) for _ in 1:100_000];
    @test ps ≈ mean(reps) atol=1e-02
end

