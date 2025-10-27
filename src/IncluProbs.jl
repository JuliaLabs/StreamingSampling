# Inclusion probabilities
# Transform weights into first order inclusion probabilities
# 1: p(w)=w*x+y
# 2: Sum constraint: sum(probs)=n => sum(p.(ws))=n
# 3: Minimum probability constraint: p(w_min)=w_min
#
# Algebra: y = w_min*(1 - x)
#          x = (n - N*w_min) / (sum(ws) - N*w_min)  == (n - N*w_min) / (N*(μ - w_min))
function inclusion_prob(sampler::Sampler, n::Integer; highprec::Bool=false)
    @views ws = sampler.weights
    N = length(ws)
    @assert N > 0 "empty weights"

    if !highprec
        wmin = minimum(ws)
        μ    = mean(ws)
        den  = N*(μ - wmin)
        num  = n - N*wmin

        # If nearly singular, switch to high precision automatically
        if den == 0 || abs(den) ≤ 1e-12 * max(1.0, abs(N*μ))
            return inclusion_prob(sampler, n; highprec=true)
        end

        x = num / den
        y = muladd(-wmin, x, wmin)              # y = wmin*(1 - x)
        ps = @. muladd(x, ws, y)                # p = x*ws + y

        return ps
    else
        # Compute coefficients in high precision, then cast back
        ps = let
            setprecision(256) do
                W    = big.(ws)
                NB   = big(N)
                nB   = big(n)
                wmin = minimum(W)
                μ    = mean(W)
                den  = NB*(μ - wmin)
                num  = nB - NB*wmin
                @assert den != 0 "Constraints are singular: mean(ws) == w_min"

                x = num / den
                y = wmin*(1 - x)
                T = eltype(ws)
                T.( @. x*W + y )
            end
        end
        return ps
    end
end
