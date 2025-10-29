# Sampling functions

# Main sampling function
function sample(sampler::Sampler, n::Int64; replace=false)
    probs = inclusion_prob(sampler, n)
    return sample(sampler, n, probs; replace=replace)
end

# UPmaxentropy sampling
function sample(sampler::StreamMaxEnt, n, probs; replace=false)
    return findall(UPmaxentropy(probs) .== 1)
end

# Weighted sampling
function sample(sampler::StreamWeights, n, probs; replace=false)
    return StatsBase.sample(1:length(probs), Weights(probs), n;
                            replace=replace)
end

