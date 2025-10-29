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
        sampler.weights = compute_weights(file_paths;
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
        sampler.weights = compute_weights(A;
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



