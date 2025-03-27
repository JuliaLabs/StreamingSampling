# Large Scale DPP approximation

mutable struct LSDPP <: Sampler
    weights::Vector{Float64}
    chunksize::Int

    function LSDPP(file_paths::Vector{String}; chunksize=200, buffersize=32,
                   max=Inf, randomized=true)
        lsdpp = new(Vector{Float64}(), chunksize)
        lsdpp.weights = compute_weights(lsdpp, file_paths; chunksize=chunksize,
                                        buffersize=buffersize, max=max,
                                        randomized=true)
        return lsdpp
    end
    
    function LSDPP(A::Matrix; chunksize=200, buffersize=32,
                   max=Inf, randomized=true)
        lsdpp = new(Vector{Float64}(), chunksize)
        lsdpp.weights = compute_weights(lsdpp, A; chunksize=chunksize,
                                        buffersize=buffersize, max=max,
                                        randomized=randomized)
        return lsdpp
    end
end

# Compute feature weights based on DPP inclusion probabilities
function compute_weights(sampler::LSDPP, features::Matrix{Float64})
    # Get number of features
    N, _ = size(features)
    # Compute pairwise Euclidean distances on the transposed features
    K = pairwise(Distances.Euclidean(), features')
    # Form an L-ensemble based on the kernel matrix K
    dpp = EllEnsemble(K)
    # Scale so that the expected size is 1
    rescale!(dpp, 1)
    # Compute inclusion probabilities.
    inclusion_probs = Determinantal.inclusion_prob(dpp)
    return inclusion_probs
end


