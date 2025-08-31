# Weight computation: used to compute inclusion probabilities and sampling

function compute_weights(sampler::Sampler, file_paths::Vector{String};
                         chunksize=1000, subchunksize=100, buffersize=32,
                         max=Inf, randomized=true)
    ch = chunk_iterator(file_paths; chunksize=subchunksize, 
                        buffersize=buffersize, randomized=randomized)
    return compute_weights(sampler, ch; chunksize=chunksize,
                           subchunksize=subchunksize, max=max)
end

function compute_weights(sampler::Sampler, A::Matrix; chunksize=1000,
                         subchunksize=100, buffersize=32, max=Inf, randomized=true)
    ch = chunk_iterator(A; chunksize=subchunksize, buffersize=buffersize,
                        randomized=randomized)
    return compute_weights(sampler, ch; chunksize=chunksize,
                           subchunksize=subchunksize, max=max)
end

function compute_weights(sampler::Sampler, ch::Channel;
                         chunksize=1000, subchunksize=100, max=Inf)
    # Step 1: Setup stage ######################################################
    @printf("Computing sampler weights...\n")
    @printf("Iteration 1. ")
    # Load elements and its global indexes
    elems = []
    ginds = []
    for _ in 1:chunksize÷subchunksize
        el, i = take!(ch)
        push!(elems, el)
        push!(ginds, i)
    end
    elems = vcat(elems...)
    ginds = vcat(ginds...)
    # Compute a feature for each element
    fs = create_features(elems)
    # Compute a weight for each feature
    ws = compute_weights(sampler, fs)
    # Allocate global weights
    gws = fill(-1.0, max) # error if max is Inf, use ch.data[ch.n_avail_items][2][end] or similar
    gws[ginds] .= ws
    # Advance number of processed elements
    nelems = length(ginds)
    @printf("No. of processed elements: %d.\n", nelems)
    
    # Step 2: Data streaming stage #############################################
    iteration = 2
    while nelems<max && isready(ch)
        @printf("Iteration %d. ", iteration)
        # Load new elements
        new_elems, new_ginds = take!(ch)
        new_elems_size = size(new_elems, 1)
        
        # Select indexes to be updated
        inds_to_update = shuffle(1:chunksize)[1:new_elems_size]
        #inds_to_update = partialsortperm(ws, 1:new_elems_size)
        
        # Update global indexes
        ginds[inds_to_update] .= new_ginds
        
        # Update elements
        elems[inds_to_update] .= new_elems
        
        # Update features: compute a feature for each new element
        fs[inds_to_update, :] .= create_features(new_elems)
        
        # Update weights: compute a weight for each feature
        ws = compute_weights(sampler, fs)
        
        # Compute functions to impute values for merging process
        ws1 = Float64[]; ws2 = Float64[];
        for i in 1:max
            if gws[i] > 0 && i in ginds
                j = findfirst(==(i), ginds)
                push!(ws1, gws[i])
                push!(ws2, ws[j])
            end
        end
        W = [ws1.^3 ws1.^2 ws1 ones(length(ws1))]
        a1, b1, c1, d1 = W \ ws2
        alpha(x) = a1*x^3+b1*x^2+c1*x+d1
        W = [ws2.^3 ws2.^2 ws2 ones(length(ws2))]
        a2, b2, c2, d2 = W \ ws1
        beta(x) = a2*x^3+b2*x^2+c2*x+d2

        # Merge
        for i in 1:max
            w1 = w2 = -1.0
            if gws[i] > -1.0
                w1 = gws[i]
                if i in ginds
                    j = findfirst(==(i), ginds)
                    w2 = ws[j]
                else
                    w2 = alpha(gws[i])
                end
            else
                if i in ginds
                    j = findfirst(==(i), ginds)
                    w1 = beta(ws[j])
                    w2 = ws[j]
                end
            end
            gws[i] = (w1+w2)/2
        end
        
        # Advance number of processed elements and iterations
        nelems += length(new_ginds)
        iteration += 1
        @printf("No. of processed elements: %d.\n", nelems)
    end
    
    @printf("Processing complete. Weights: %d\n", length(gws))
    return gws
end
