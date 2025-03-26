using PotentialLearning
using OrderedCollections

# LSSampling
include("../../src/lssampling.jl")

# Domain specific functions
include("utils/macros.jl")
include("utils/samplers.jl")
include("utils/aux_sample_functions.jl")
include("utils/plots.jl")
include("utils/plotmetrics.jl")
include("utils/atom-conf-features-extxyz.jl")
include("utils/subtract_peratom_e.jl")

# Data #########################################################################

# Define paths and create experiment folder
res_path  = "results-iso17/"
run(`mkdir -p $res_path`)

# Load training atomistic configurations (random subset of size N)
N = 200
file_paths = ["data/iso17/my_iso17_train.extxyz"]
ch = chunk_iterator(file_paths; chunksize=N)
chunk, _ = take!(ch)
confs = []
for (system, energy, forces) in chunk
    conf = Configuration(system, Energy(energy),
           Forces([Force(f) for f in forces]))
    push!(confs, conf)
end
confs = DataSet(confs)

# For ISO17, all confs have the same number of atoms
num_atoms = length(get_system(ds_train[1]))
all_energies = get_all_energies(confs)
avg_energy_per_atoms = mean(all_energies)/num_atoms 
vref_dict = Dict(:H => avg_energy_per_atom,
                 :C => avg_energy_per_atom,
                 :O => avg_energy_per_atom) 

# This will permanently change the energies in the entire dataset !!!
adjust_energies(confs,vref_dict)

# Define basis
basis = ACE(species           = [:C, :O, :H],
            body_order        = 4,
            polynomial_degree = 12,
            wL                = 2.0,
            csp               = 1.0,
            r0                = 1.43,
            rcutoff           = 4.4 );

# Update training dataset by adding energy and force descriptors
#println("Computing energy descriptors of dataset...")
#B_time = @elapsed e_descr = compute_local_descriptors(confs, basis)
#println("Computing force descriptors of dataset...")
#dB_time = @elapsed f_descr = compute_force_descriptors(confs, basis)
#GC.gc()
#ds_train = DataSet(confs .+ e_descr .+ f_descr)
ds_train = deepcopy(confs)

# Load test atomistic configurations (random subset of size N)
M = 100
file_paths = ["data/iso17/my_iso17_test.extxyz"]
ch = chunk_iterator(file_paths; chunksize=M)
chunk, _ = take!(ch)
confs = []
for (system, energy, forces) in chunk
    conf = Configuration(system, Energy(energy),
           Forces([Force(f) for f in forces]))
    push!(confs, conf)
end
confs = DataSet(confs)

# Note, I don't modify the test set energies !!!

# Update test dataset by adding energy and force descriptors
#println("Computing energy descriptors of dataset...")
#B_time = @elapsed e_descr = compute_local_descriptors(confs, basis)
#println("Computing force descriptors of dataset...")
#dB_time = @elapsed f_descr = compute_force_descriptors(confs, basis)
#GC.gc()
#ds_test = DataSet(confs .+ e_descr .+ f_descr)
ds_test = deepcopy(confs)


# Sampling experiments #########################################################

# Define number of experiments
n_experiments = 40

# Define samplers
#samplers = [simple_random_sample, dbscan_sample, kmeans_sample, 
#            cur_sample, dpp_sample, lrdpp_sample]
samplers = [simple_random_sample, kmeans_sample, cur_sample, dpp_sample]

# Define batch sample sizes (proportions)
#batch_size_props = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
batch_size_props = [0.08, 0.16, 0.32, 0.64]

# Create metric dataframe
metric_names = [:exp_number,  :method, :batch_size_prop, :batch_size, :time,
                :e_train_mae, :e_train_rmse, :e_train_rsq,
                :f_train_mae, :f_train_rmse, :f_train_rsq, :f_train_mean_cos,
                :e_test_mae,  :e_test_rmse,  :e_test_rsq, 
                :f_test_mae,  :f_test_rmse,  :f_test_rsq,  :f_test_mean_cos]
metrics = DataFrame([Any[] for _ in 1:length(metric_names)], metric_names)

# Run experiments
for j in 1:n_experiments
    global metrics
    
    # Define randomized training and test dataset
    # A randomized 80% of each dataset is used in each experiment
    n_train = floor(Int, 0.8*length(ds_train))
    n_test = floor(Int, 0.8*length(ds_test))
    inds_train = randperm(length(ds_train))[1:n_train]
    inds_test = randperm(length(ds_test))[1:n_test]
    ds_train_rnd = @views ds_train[inds_train]
    ds_test_rnd  = @views ds_test[inds_test]
    ged = sum.(get_values.(get_local_descriptors.(ds_train_rnd)))
    ged_mat = stack(ged)'
    
    # Sampling experiments
    for batch_size_prop in batch_size_props
        for sampler in samplers
            sample_experiment!(res_path, j, sampler, batch_size_prop, n_train, 
                               ged_mat, ds_train_rnd, ds_test_rnd, basis, metrics; vref_dict=vref_dict)
            GC.gc()
        end
    end
end

# Postprocess ##################################################################
plotmetrics(res_path, "metrics.csv")

