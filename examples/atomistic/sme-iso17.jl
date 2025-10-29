#using Pkg
#Pkg.develop(path="../../")

using StreamingSampling
using StatsBase

include("utils/utils.jl")

# Datase files
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
function create_feature(element::Vector; basis=basis)
    system = element[1]
    feature = sum(compute_local_descriptors(system, basis))
    return feature
end

# Compute streaming weights
ws = compute_weights(file_paths;
                     read_element=read_element,
                     create_feature=create_feature,
                     chunksize=2000,
                     subchunksize=200)

# Define sample size
n = 100

# Sample by weighted sampling
inds = StatsBase.sample(1:length(ws), Weights(ws), n; replace=false)

