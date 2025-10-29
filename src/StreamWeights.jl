# Streaming Weights

mutable struct StreamWeights
    weights::Vector{Float64}
    chunksize::Int
    subchunksize::Int

    function StreamWeights(file_paths::Vector{String}; 
                              read_element=read_element,
                              create_feature=create_feature,
                              chunksize=2000,
                              subchunksize=200,
                              buffersize=32,
                              max=Inf,
                              randomized=true)
        ws = new(Vector{Float64}(),
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
        return ws
    end
    
    function StreamWeights(A::Vector;
                              read_element=read_element,
                              create_feature=create_feature,
                              chunksize=2000,
                              subchunksize=200,
                              buffersize=32,
                              max=Inf,
                              randomized=true)
        ws = new(Vector{Float64}(),
                 chunksize,
                 subchunksize)
        ws.weights = compute_weights(sampler, A;
                                     read_element=read_element,
                                     create_feature=create_feature,
                                     chunksize=chunksize,
                                     subchunksize=subchunksize,
                                     buffersize=buffersize,
                                     max=max,
                                     randomized=randomized)
        return ws
    end
end



