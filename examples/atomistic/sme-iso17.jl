using StreamingSampling

# Domain specific functions
include("utils/AtomsSampling.jl")
include("utils/atom-conf-features-extxyz.jl") # read_element
basis = ACE(species           = [:C, :O, :H],
            body_order        = 4,
            polynomial_degree = 12,
            wL                = 2.0,
            csp               = 1.0,
            r0                = 1.43,
            rcutoff           = 4.4 );
function create_feature(element::Vector)
    system = element[1]
    feature = sum(compute_local_descriptors(system, basis))
    return feature
end

# Sampling
file_paths = ["data/iso17/my_iso17_train.extxyz"] 
sme = StreamMaxEnt(file_paths; chunksize=2000, subchunksize=200)
inds = sample(sme, 10)

