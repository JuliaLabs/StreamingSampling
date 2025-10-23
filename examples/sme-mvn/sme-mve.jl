using StreamingSampling

include("gen-data.jl")

# Generate synthetic data
file_paths = ["data1.txt", "data2.txt", "data3.txt", "data4.txt"]
generate_data(file_paths)

# Sampling by StreamMaxEnt
sme = StreamMaxEnt(file_paths; chunksize=200)
inds = sample(sme, 100)

