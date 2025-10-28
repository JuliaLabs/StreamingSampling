using StreamingSampling

# Define file paths
base = haskey(ENV, "BASE_PATH") ? ENV["BASE_PATH"] : "../../"
path = "$base/test/data/"
file_paths = ["$path/data1.txt",
              "$path/data2.txt",
              "$path/data3.txt",
              "$path/data4.txt"]

# Sample by StreamMaxEnt
sme = StreamMaxEnt(file_paths; chunksize=500, subchunksize=100)
n = 100
inds = sample(sme, n)

# Get first-order inclusion probabilities for a sample size n 
ps = inclusion_prob(sme, n)

# Compute summation of inclusion probabilities
s = round(Int, sum(ps))

# Check sample size
length(inds) == s
    
# Check sum(ps) == n
s == n
    
# Check 0 <= p_i <= 1
all(0 .<= ps .<= 1)

