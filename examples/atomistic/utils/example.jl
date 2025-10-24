"""
Example of how the future `AtomsSampling.jl` package may be used.
"""

using AtomsSampling

# Dataset files
file_paths = ["data/iso17/my_iso17_train.extxyz"]

# Dataset-specific functions
read_element(io) = read_element_extxyz(io)
basis = ACE(species           = [:C, :O, :H],
            body_order        = 4,
            polynomial_degree = 12,
            wL                = 2.0,
            csp               = 1.0,
            r0                = 1.43,
            rcutoff           = 4.4 );
function create_feature(element::Vector, basis=basis)
    system = element[1]
    feature = sum(compute_local_descriptors(system, basis))
    return feature
end

# Create sampler
sampler = StreamMaxEnt(file_paths;
                       read_element=read_element,
                       create_feature=creature_feature)

# Sample
inds = sample(sampler, 10)

