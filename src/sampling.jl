# Sampling functions

# Main sampling function
function sample(sampler::Sampler, n::Int64; replace=false)
    probs = inclusion_prob(sampler, n)
    return sample(probs, n)
end
#function sample(sampler::Sampler, n::Int64; replace=false)
#    return sample(sampler.weights, n)
#end


# Conditional poisson sampling
#function sample(p::Vector{Float64}, n::Int)
#    N = length(p)
#    while true
#        S = Int[]
#        @inbounds for i in 1:N
#            if rand() < p[i]; push!(S, i); end
#        end
#        if length(S) == n
#            return S
#        end
#    end
#end

# Sampling function inspired by conditional poisson sampling
function sample(probabilities::Vector{Float64}, n::Int)

    N = length(probabilities)

    # Crucial check: Verify sum of probabilities
    if abs(sum(probabilities) - n) > 1e-6  # Using a tolerance
        error("Sum of probabilities must equal n. Sum: $(sum(probabilities)), n: $(n)")
    end

    # 1. Poisson Sampling
    poisson_sample = Int[]
    for i in 1:N
        if rand() < probabilities[i]
            push!(poisson_sample, i)
        end
    end

    # 2. Conditioning
    sample_size = length(poisson_sample)

    if sample_size > n
        # Remove elements
        num_to_remove = sample_size - n
        for _ in 1:num_to_remove
            remove_idx = rand(1:length(poisson_sample))
            deleteat!(poisson_sample, remove_idx)
        end
    elseif sample_size < n
        # Add elements
        num_to_add = n - sample_size
        remaining_indices = setdiff(1:N, poisson_sample) # Indices not yet selected
        remaining_probabilities = probabilities[remaining_indices]
        
        for _ in 1:num_to_add
          
            total_prob = sum(remaining_probabilities)
            if total_prob == 0.0
                break # Handle edge case where no elements can be added
            end
            
            random_value = rand() * total_prob
            cumulative_prob = 0.0
            add_idx_idx = 0
            for (i, prob) in enumerate(remaining_probabilities)
                cumulative_prob += prob
                if random_value <= cumulative_prob
                    add_idx_idx = i
                    break
                end
            end
            
            push!(poisson_sample, remaining_indices[add_idx_idx])
            deleteat!(remaining_indices, add_idx_idx)
            deleteat!(remaining_probabilities,add_idx_idx)
        end
    end

    return poisson_sample
end

# Weighted sampling
#function sample(probabilities::Vector{Float64}, n::Int; replace=false)
#    return StatsBase.sample(collect(1:length(probabilities)), 
#                            Weights(probabilities), n; replace=replace)
#end
