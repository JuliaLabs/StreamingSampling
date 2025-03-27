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

# Data #########################################################################

# Define paths and create experiment folder
res_path  = "results-hf/"
run(`mkdir -p $res_path`)

# Load atomistic configurations (random subset of size N)
N = 5000
file_paths = ["data/Hf/Hf2_gas_form_sorted.extxyz",
              "data/Hf/Hf2_mp103_EOS_1D_form_sorted.extxyz",
              "data/Hf/Hf2_mp103_EOS_3D_form_sorted.extxyz",
              "data/Hf/Hf2_mp103_EOS_6D_form_sorted.extxyz",
              "data/Hf/Hf128_MC_rattled_mp100_form_sorted.extxyz",
              "data/Hf/Hf128_MC_rattled_mp103_form_sorted.extxyz",
              "data/Hf/Hf128_MC_rattled_random_form_sorted.extxyz",
              "data/Hf/Hf_mp100_EOS_1D_form_sorted.extxyz",
              "data/Hf/Hf_mp100_primitive_EOS_1D_form_sorted.extxyz"]
ch = chunk_iterator(file_paths; chunksize=N)
chunk, _ = take!(ch)
confs = []
for (system, energy, forces) in chunk
    conf = Configuration(system, Energy(energy),
           Forces([Force(f) for f in forces]))
    push!(confs, conf)
end
confs = DataSet(confs)

# Define basis
basis = ACE(species           = [:Hf],
            body_order        = 6,
            polynomial_degree = 8,
            rcutoff           = 5.0,
            wL                = 1.0,
            csp               = 1.0,
            r0                = 1.0);

# Update dataset by adding energy and force descriptors
println("Computing energy descriptors of dataset...")
B_time = @elapsed e_descr = compute_local_descriptors(confs, basis)
println("Computing force descriptors of dataset...")
dB_time = @elapsed f_descr = compute_force_descriptors(confs, basis)
GC.gc()
ds = DataSet(confs .+ e_descr .+ f_descr)


# Sampling experiments #########################################################

# Define number of experiments
n_experiments = 40

# Define samplers
#samplers = [simple_random_sample, dbscan_sample, kmeans_sample, 
#            cur_sample, dpp_sample, lrdpp_sample]
samplers = [simple_random_sample, kmeans_sample, cur_sample, dpp_sample, lsdpp_sample]

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
    n_train = floor(Int, 0.8*length(ds))
    inds_train = randperm(length(ds))[1:n_train]
    inds_test = randperm(length(ds))[n_train+1:end]
    ds_train_rnd = @views ds[inds_train]
    ds_test_rnd  = @views ds[inds_test]
    ged = sum.(get_values.(get_local_descriptors.(ds_train_rnd)))
    ged_mat = stack(ged)'
    
    # Sampling experiments
    for batch_size_prop in batch_size_props
        for curr_sampler in samplers
            sample_experiment!(res_path, j, curr_sampler, batch_size_prop, n_train, 
                               ged_mat, ds_train_rnd, ds_test_rnd, basis, metrics)
            GC.gc()
        end
    end
end

# Postprocess ##################################################################
plotmetrics(res_path, "metrics.csv")

