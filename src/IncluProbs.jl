# Inclusion probabilities
# Transform weights into first order inclusion probabilities
# 1: p(w)=x*w+y
# 2: Sum constraint: sum(probs)=n => sum(p.(ws))=n
# 3: Minimum probability constraint: p(w_min)=w_min
function inclusion_prob(sampler::Sampler, n::Int)
    @views ws = sampler.weights
    N = length(ws)
    min_prob = minimum(ws)
    A = [ sum(ws) N;
          minimum(ws) 1.0]
    b = [n, min_prob]
    x, y = A \ b
    ps = x*ws.+y
    return ps
end

