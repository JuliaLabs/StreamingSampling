# LSSampling
include("../../src/StreamingSampling.jl")

# Sample size and dataset size
n = 200
N = 4000

# Generate synthetic data
file_paths = ["data1.txt", "data2.txt", "data3.txt", "data4.txt"]
generate_data(file_paths; N=N, feature_size=50);

# Sampling by DPP
Random.seed!(42) # Fixed seed to compare DPP and StreamMaxEnt: get same random chunks
@time begin
    ch, _ = chunk_iterator(file_paths; chunksize=N)
    chunk, _ = take!(ch)
    features = create_features(chunk)
    D = pairwise(Euclidean(), features')
    K = exp.(-D.^2)
    dpp = EllEnsemble(K)
    rescale!(dpp, n)
    dpp_probs = Determinantal.inclusion_prob(dpp)
    dpp_indexes = Determinantal.sample(dpp, n)
end
chunk = nothing;
features = nothing;
GC.gc()

# Sampling by StreamMaxEnt
Random.seed!(42) # Fix seed to compare DPP and StreamMaxEnt: get same random chunks
@time begin
    sme = StreamMaxEnt(file_paths; chunksize=500, max=N)
    sme_probs = inclusion_prob(sme, n)
    sme_indexes = sample(sme, n)
end

# Tests and plots

# DPP vs. StreamMaxEnt inclusion probabilities when sampling 
# n points from a set of size N, with each point of size M
scatter(dpp_probs, sme_probs, color="red", alpha=0.5)
plot!(dpp_probs, dpp_probs, color="blue", alpha=0.5)
plot!(xlabel="DPP inclusion probabilities")
plot!(ylabel="StreamMaxEnt inclusion probabilities")
plot!(legend=false, dpi=300)
savefig("dpp-probs-vs-sme-probs-mvn.png")

# DPP theoretical inclusion probabilities vs StreamMaxEnt inclusion frequencies when
# sampling n points from a set of size N, with each point of size M
iterations = 20_000_000 # Use 20_000_000
sme_freqs = relative_frequencies(sme, n, iterations)
scatter(dpp_probs, sme_freqs, color="red", alpha=0.5)
plot!(dpp_probs, dpp_probs, color="blue", alpha=0.5)
plot!(xlabel="DPP inclusion probabilities")
plot!(ylabel="StreamMaxEnt inclusion frequencies")
plot!(legend=false, dpi=300)
savefig("dpp-probs-vs-sme-freqs-mvn.png")

# DPP theoretical inclusion probabilities vs StreamMaxEnt inclusion frequencies of 2 
# random points, when sampling n points from a set of size N, with each point of size M
set = rand(1:N, 2)
iterations = 1_000_000
sme_set_freqs = relative_frequencies(sme, set, n, iterations)
dpp_set_freqs = det(marginal_kernel(dpp)[set, set])
@printf("DPP inclusion probability for dataset %s is %f \n", 
         string(set), dpp_set_freqs)
@printf("StreamMaxEnt inclusion probability for dataset %s is %f \n",
         string(set), sme_set_freqs)

