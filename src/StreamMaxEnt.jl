# Streaming Maximum Entropy Sampling

mutable struct StreamMaxEnt <: Sampler
    weights::Vector{Float64}
    chunksize::Int
    subchunksize::Int

    function StreamMaxEnt(file_paths::Vector{String}; 
                          read_element=read_element,
                          create_feature=create_feature,
                          chunksize=2000,
                          subchunksize=200,
                          buffersize=32,
                          max=Inf,
                          randomized=true)
        sampler = new(Vector{Float64}(),
                      chunksize,
                      subchunksize)
        sampler.weights = compute_weights(sampler, file_paths;
                                          read_element=read_element,
                                          create_feature=create_feature,
                                          chunksize=chunksize,
                                          subchunksize=subchunksize,
                                          buffersize=buffersize,
                                          max=max,
                                          randomized=randomized)
        return sampler
    end
    
    function StreamMaxEnt(A::Vector;
                          read_element=read_element,
                          create_feature=create_feature,
                          chunksize=2000,
                          subchunksize=200,
                          buffersize=32,
                          max=Inf,
                          randomized=true)
        sampler = new(Vector{Float64}(),
                      chunksize,
                      subchunksize)
        sampler.weights = compute_weights(sampler, A;
                                          read_element=read_element,
                                          create_feature=create_feature,
                                          chunksize=chunksize,
                                          subchunksize=subchunksize,
                                          buffersize=buffersize,
                                          max=max,
                                          randomized=randomized)
        return sampler
    end
end

# Compute feature weights based on DPP inclusion probabilities
function compute_weights(sampler::StreamMaxEnt,
                         features::Matrix{Float64})
    # Get number of features
    N, _ = size(features)
    # Compute pairwise Euclidean distances on the transposed features
    D = pairwise(Distances.Euclidean(), features')
    K = exp.(-D.^2)
    # Form an L-ensemble based on the kernel matrix K
    dpp = EllEnsemble(K)
    # Scale so that the expected size is 1
    rescale!(dpp, 1)
    # Compute inclusion probabilities.
    inclusion_probs = Determinantal.inclusion_prob(dpp)
    return inclusion_probs
end


