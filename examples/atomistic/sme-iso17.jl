using StreamingSampling

# Domain specific functions
include("utils/AtomsSampling.jl")

# Basis function to compute ACE descriptors (features)
basis = ACE(species           = [:C, :O, :H],
            body_order        = 4,
            polynomial_degree = 12,
            wL                = 2.0,
            csp               = 1.0,
            r0                = 1.43,
            rcutoff           = 4.4 );

# Dataset
file_paths = ["data/iso17/my_iso17_train.extxyz"] 

# Sample by StreamMaxEnt
sme = StreamMaxEnt(file_paths; chunksize=2000, subchunksize=200)
sme_indexes = sample(sme, 10)
serialize(sme, "sme.jls")
