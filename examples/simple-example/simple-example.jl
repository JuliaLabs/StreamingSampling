using StreamingSampling
using StatsBase

# Define file paths
base = haskey(ENV, "BASE_PATH") ? ENV["BASE_PATH"] : "../../"
path = "$base/test/data/"
file_paths = ["$path/data1.txt",
              "$path/data2.txt",
              "$path/data3.txt",
              "$path/data4.txt"];

# Compute streaming weights
ws = compute_weights(file_paths; chunksize=500, subchunksize=100)

# Define sample size
n = 100;

# Compute first-order inclusion probabilities with sample size n
ps = inclusion_prob(ws, n)

# Option 1: Sample by weighted sampling
inds_w = StatsBase.sample(1:length(ws), Weights(ws), n; replace=false)

# Option 2: Sample by weighted sampling and first-order inclusion probabilities
inds_p = StatsBase.sample(1:length(ps), Weights(ps), n; replace=false)

# Option 3: Sample by UPmaxentropy
s = UPmaxentropy(ps)
inds_me = findall(s .== 1)

