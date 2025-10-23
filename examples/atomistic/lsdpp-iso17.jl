# LSSampling
include("../../src/lssampling.jl")

# Domain specific feature calculation
include("utils/atom-conf-features-extxyz.jl")

# Basis function to compute ACE descriptors (features)
basis = ACE(species           = [:C, :O, :H],
            body_order        = 4,
            polynomial_degree = 4,
            rcutoff           = 5.0,
            wL                = 1.0,
            csp               = 1.0,
            r0                = 1.0);

# Dataset
file_paths = ["data/iso17/my_iso17_train.extxyz"] 

# Sample by LSDPP
lsdpp = LSDPP(file_paths; chunksize=2000, subchunksize=200)
lsdpp_indexes = sample(lsdpp, 10)
serialize(lsdpp, "lsdpp.jls")

s100 = sample(lsdpp, 100)
serialize("s100.jls", s100)

lsdpp_probs = inclusion_prob(lsdpp, 100)

s1000 = sample(lsdpp, 1000)
serialize("s1000.jls", s1000)

s10000 = sample(lsdpp, 10000)
serialize("s10000.jls", s10000)

lsdpp_probs = inclusion_prob(lsdpp, 200)
