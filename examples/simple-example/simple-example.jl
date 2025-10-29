using StreamingSampling
using StatsBase: sample, Weights
using Plots

# Define file paths
base = haskey(ENV, "BASE_PATH") ? ENV["BASE_PATH"] : "../../"
path = "$base/test/data/"
file_paths = ["$path/data1.txt",
              "$path/data2.txt",
              "$path/data3.txt",
              "$path/data4.txt"];

include("$base/examples/simple-example/plot_weights.jl"); #hide

# Define sample size
n = 30;

# Streaming weighted sampling
ws = compute_weights(file_paths; chunksize=500, subchunksize=100)
inds_w = sample(1:length(ws), Weights(ws), n; replace=false)
plot_weights(ws, inds_w) #hide

# Streaming maximum entropy sampling
s = UPmaxentropy(inclusion_prob(ws, n))
inds_me = findall(s .== 1)
plot_weights(ws, inds_me) #hide
