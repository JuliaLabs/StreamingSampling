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
lsdpp_indexes = sample(lsdpp, n)

