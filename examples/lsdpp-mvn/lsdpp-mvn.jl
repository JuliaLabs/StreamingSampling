include("../../src/lssampling.jl")

# Generate synthetic data
file_paths = ["data1.txt", "data2.txt", "data3.txt", "data4.txt"]
generate_data(file_paths)

# Sampling by LSDPP
lsdpp = LSDPP(file_paths; chunksize=200)
inds = sample(lsdpp, 100)

