# LSSampling
include("../../src/lssampling.jl")

# Domain specific feature calculation
include("utils/atom-conf-features-extxyz.jl")

# Basis function to compute ACE descriptors (features)
basis = ACE(species           = [:C, :O, :H],
            body_order        = 4,
            polynomial_degree = 6,
            wL                = 2.0,
            csp               = 1.0,
            r0                = 1.43,
            rcutoff           = 4.4 );

# Data
path = ["data/iso17/my_iso17_train.extxyz"]
res_path  = "results-probs-iso17/"
run(`mkdir -p $res_path`)

# Sample size and dataset size (the dataset is a sample of the full dataset)
n = 200
m = 5000

# Compute features
ch, N = chunk_iterator(path; chunksize=m)
structs = first(take!(ch))
close(ch); GC.gc()
features = create_features(structs)

# Sampling by DPP
D = pairwise(Euclidean(), features')
K = exp.(-D.^2)
dpp = EllEnsemble(K)
rescale!(dpp, n)
dpp_probs = Determinantal.inclusion_prob(dpp)
dpp_indexes = Determinantal.sample(dpp, n)

# Sampling by LSDPP
lsdpp = LSDPP(structs; chunksize=2000, subchunksize=200, randomized=false)
lsdpp_probs = inclusion_prob(lsdpp, n)
lsdpp_indexes = sample(lsdpp, n)

# Tests and plots

# DPP vs. LSDPP inclusion probabilities when sampling 
# n points from a set of size N, with each point of size M
scatter(dpp_probs, lsdpp_probs, color="red", alpha=0.5)
plot!(dpp_probs, dpp_probs, color="blue", alpha=0.5)
plot!(xlabel="DPP inclusion probabilities")
plot!(ylabel="LSDPP inclusion probabilities")
plot!(legend=false, dpi=300)
savefig("$res_path/dpp-probs-vs-lsdpp-probs-iso17-1.png")

plot(dpp_probs, color="red", alpha=0.5, label="DPP inclusion probabilities")
plot!(lsdpp_probs, color="blue", alpha=0.5, label="LSDPP inclusion probabilities")
plot!(xlabel="Structures", ylabel="Probability", legend=:bottomright)
savefig("$res_path/dpp-probs-vs-lsdpp-probs-iso17-2.png")

inds = sortperm(dpp_probs)
plot(dpp_probs[inds], color="red", alpha=0.5, label="DPP inclusion probabilities")
plot!(lsdpp_probs[inds], color="blue", alpha=0.5, label="LSDPP inclusion probabilities")
plot!(xlabel="Structure sorted by DPP probabilities", ylabel="Probability", legend=:bottomright)
savefig("$res_path/dpp-probs-vs-lsdpp-probs-iso17-sorted.png")

plot(cumsum(dpp_probs), color="red", alpha=0.5, label="DPP inclusion probabilities")
plot!(cumsum(lsdpp_probs), color="blue", alpha=0.5, label="LSDPP inclusion probabilities")
plot!(xlabel="Structure", ylabel="Probability", legend=:bottomright)
savefig("$res_path/dpp-probs-vs-lsdpp-probs-iso17-cumsum.png")

# DPP theoretical inclusion probabilities vs LSDPP inclusion frequencies when
# sampling n points from a set of size N, with each point of size M
iterations = 100_000
lsdpp_freqs = relative_frequencies(lsdpp, n, iterations)
scatter(dpp_probs, lsdpp_freqs, color="red", alpha=0.5)
plot!(dpp_probs, dpp_probs, color="blue", alpha=0.5)
plot!(xlabel="DPP inclusion probabilities")
plot!(ylabel="LSDPP inclusion frequencies")
plot!(legend=false, dpi=300)
savefig("$res_path/dpp-probs-vs-lsdpp-freqs-iso17.png")

# DPP theoretical inclusion probabilities vs LSDPP inclusion frequencies of 2 
# random points, when sampling n points from a set of size N, with each point of size M
set = rand(1:m, 2)
iterations = 100_000
lsdpp_set_freqs = relative_frequencies(lsdpp, set, n, iterations)
dpp_set_freqs = det(marginal_kernel(dpp)[set, set])
@printf("DPP inclusion probability for dataset %s is %f \n", 
         string(set), dpp_set_freqs)
@printf("LSDPP inclusion probability for dataset %s is %f \n",
         string(set), lsdpp_set_freqs)

