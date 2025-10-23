# LSSampling
include("../../src/StreamingSampling.jl")

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

# Sample by StreamMaxEnt
sme = StreamMaxEnt(file_paths; chunksize=2000, subchunksize=200)
sme_indexes = sample(sme, 10)
serialize(sme, "sme.jls")

s100 = sample(sme, 100)
serialize("s100.jls", s100)

sme_probs = inclusion_prob(sme, 100)

s1000 = sample(sme, 1000)
serialize("s1000.jls", s1000)

s10000 = sample(sme, 10000)
serialize("s10000.jls", s10000)

sme_probs = inclusion_prob(sme, 200)
