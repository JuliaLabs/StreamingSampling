# Inclusion probabilities and relative frequencies functions

# Transform weights into inclusion probabilities
function inclusion_prob(sampler::Sampler, n::Int)
# 1) Sum constraint: sum(probs)=n => sum(x*ws.+y)=n
# 2) Minimum probability constraint: 
#      x*minimum(ws)+y= n/N - 1/12000 (empirical constraint)
#    @views ws = sampler.weights
#    N = length(ws)
#    A = [ sum(ws) N;
#          minimum(ws) 1.0]
#    min_prob = n/N - 1/12000
#    b = [n, min_prob]
#    x, y = A \ b
#    return x*ws.+y
    
    @views ws = sampler.weights
    # Squashing function h(t)∈(0,1)
    h(t) = 1-exp(-t)
    # Find a scale α>0 so that the sum hits the target
    f(α) = sum(h.(α.*ws)) - n
    α = find_zero(f, (-2n, 2n), Bisection())
    return h.(α.*ws)
end

function relative_frequencies(sampler::Sampler, n::Int, iterations::Int)
    N = length(sampler.weights)
    partial_counts = [zeros(N) for _ in 1:nthreads()]
    @threads for i in 1:iterations
        tid = threadid()
        local_counts = partial_counts[tid]
        selected_indices = sample(sampler, n)
        for idx in selected_indices
            local_counts[idx] += 1
        end
    end
    total_counts = reduce(+, partial_counts)
    return total_counts ./ iterations
end

function relative_frequencies(sampler::Sampler, set::Vector{Int},
                              n::Int, iterations::Int)
    total_counts = Threads.Atomic{Int}(0)
    @threads for i in 1:iterations
        inds = sample(sampler, n)
        if all(x -> x in inds, set)
            Threads.atomic_add!(total_counts, 1)
        end
    end
    return total_counts[] ./ iterations
end

