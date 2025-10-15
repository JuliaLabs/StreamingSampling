# Sampling functions

# Main sampling function
function sample(sampler::Sampler, n::Int64; replace=false)
    probs = inclusion_prob(sampler, n)
    return sample(probs)
end

function sample(probs)
    return findall(x -> x == 1, UPmaxentropy(probs))
end

#function sample(sampler::Sampler, n::Int64; replace=false)
#    return sample(sampler.weights, n)
#end

# Weighted sampling
#function sample(probabilities::Vector{Float64}, n::Int; replace=false)
#    return StatsBase.sample(collect(1:length(probabilities)), 
#                            Weights(probabilities), n; replace=replace)
#end
