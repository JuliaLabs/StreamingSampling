# Weight computation: used to compute inclusion probabilities and sampling

function compute_weights(sampler::Sampler, file_paths::Vector{String};
                         chunksize=200, max=Inf)

    @printf("Computing sampler weights...\n")
    ch = chunk_iterator(file_paths; chunksize=chunksize)
    
    # First iteration
    @printf("Starting first iteration...\n")
    chunk1, idx1 = take!(ch)
    chunk2, idx2 = take!(ch)
    features = create_features([chunk1; chunk2])
    _, M = size(features)
    feature_global_indexes = [idx1; idx2]
    feature_weights = compute_weights(sampler, features)
    global_weights = copy(feature_weights)
    @printf("Iteration 1 complete.  Current number of weights: %d.\n", 
            length(global_weights))
    
    # Subsequent iterations
    iteration = 2
    while length(global_weights)<max && isready(ch)
        new_chunk, new_chunk_global_indexes = take!(ch)
        @printf("Starting iteration %d...\n", iteration)
        new_chunksize = length(new_chunk)
        features_to_update = shuffle(collect(1:2chunksize))[1:new_chunksize]
        features[features_to_update, :] .= create_features(new_chunk)
        feature_global_indexes[features_to_update] .= new_chunk_global_indexes
        
        curr_feature_weights = compute_weights(sampler, features)
        for i in 1:2M
            if i in features_to_update
                feature_weights[i] = curr_feature_weights[i]
            else
                feature_weights[i] = (feature_weights[i]+curr_feature_weights[i])/2
            end
        end
        feature_weights .= curr_feature_weights
        
        append!(global_weights, zeros(new_chunksize))
        global_weights[feature_global_indexes] .= feature_weights
        @printf("Iteration %d complete. Current number of weights: %d.\n", 
                iteration, length(global_weights))
        iteration += 1
    end
    
    @printf("Processing complete. Weights: %d\n", length(global_weights)) 
    return global_weights
end

