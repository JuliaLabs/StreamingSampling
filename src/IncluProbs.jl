# Inclusion probabilities
# Transform weights into first order inclusion probabilities
# 1: p(w)=n*w/sum(w)
# 2: Sum constraint: sum(probs)=n => sum(p.(ws))=n
function inclusion_prob(sampler::Sampler, n::Int)
    @views ws = sampler.weights
    N = length(ws)
    
    # Start with probabilities proportional to weights
    probs_proportional = n * ws / sum(ws)
    
    # Check if already in [0,1] - if so, we're done!
    if all(0 .<= probs_proportional .<= 1)
        return probs_proportional
    end
    
    # Otherwise, need to adjust with optimization
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    
    # Direct variables for each probability
    @variable(model, 0 <= p[i=1:N] <= 1)
    
    # Constraint: Sum equals n
    @constraint(model, sum(p) == n)
    
    # Objective: minimize deviation from proportional probabilities
    # This maintains the relative importance from weights
    @objective(model, Min, sum((p[i] - probs_proportional[i])^2 for i in 1:N))
    
    optimize!(model)
    
    if termination_status(model) != MOI.OPTIMAL && termination_status(model) != MOI.LOCALLY_SOLVED
        error("Could not find valid probabilities. Check that 0 < n <= N=$(N)")
    end
    
    return value.(p)
end

