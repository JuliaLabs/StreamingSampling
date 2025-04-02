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
include("utils/xyz.jl")

# Data #########################################################################

# Define paths and create experiment folder
res_path  = "results-dpps-hf-balanced-partial-stats/"
run(`mkdir -p $res_path`)

# Load atomistic configurations (random subset of size N)
file_paths = ["data/Hf/Hf2_gas_form_sorted.extxyz",
              "data/Hf/Hf2_mp103_EOS_1D_form_sorted.extxyz",
              "data/Hf/Hf2_mp103_EOS_3D_form_sorted.extxyz",
              "data/Hf/Hf2_mp103_EOS_6D_form_sorted.extxyz",
              "data/Hf/Hf128_MC_rattled_mp100_form_sorted.extxyz",
              "data/Hf/Hf128_MC_rattled_mp103_form_sorted.extxyz",
              "data/Hf/Hf128_MC_rattled_random_form_sorted.extxyz",
              "data/Hf/Hf_mp100_EOS_1D_form_sorted.extxyz",
              "data/Hf/Hf_mp100_primitive_EOS_1D_form_sorted.extxyz"]

confs = []
confsizes = zeros(Int, length(file_paths))
for (i, ds_path) in enumerate(file_paths)
    newconfs = load_data(ds_path, uparse("eV"), uparse("Å"))
    push!(confs, newconfs...)
    confsizes[i] = length(newconfs)
end
offsets = zeros(Int, length(file_paths))
for i in 2:length(file_paths)
    offsets[i] = confsizes[i-1] + offsets[i-1]
end
confs = DataSet(confs)
N = length(confs)
GC.gc()

# Define basis
basis = ACE(species           = [:Hf],
            body_order        = 6,
            polynomial_degree = 10,
            rcutoff           = 5.5,
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


# Define randomized training and test dataset.
# Here, both datasets have elements of each file.
rnd_inds_train = Int[]
rnd_inds_test = Int[]
for (i, ni) in enumerate(confsizes)
    n_train_i = floor(Int, 0.8 * ni)
    n_test_i = ni - n_train_i
    rnd_inds = randperm(ni) .+ offsets[i]
    push!(rnd_inds_train, rnd_inds[1:n_train_i]...)
    push!(rnd_inds_test, rnd_inds[n_train_i+1:n_train_i+n_test_i]...)
end
ds_train_rnd = @views ds[rnd_inds_train]
ds_test_rnd  = @views ds[rnd_inds_test]
n_train = length(ds_train_rnd)
n_test = length(ds_test_rnd)
ged = sum.(get_values.(get_local_descriptors.(ds_train_rnd)))
A = stack(ged)'

# Samplers #####################################################################

# LRDPP
 # Compute a kernel matrix for the points in x
L = LowRank(Matrix(A))
# Form an L-ensemble based on the L matrix
dpp = EllEnsemble(L)
function lrdpp_sample(A, n)
    global dpp
    # Sample A (obtain indices). Use resampling if needed.
    _, N = Base.size(A)
    n′ = n > N ? N : n
    curr_n = 0
    inds = []
    it_max = 1000
    i = 0
    while curr_n < n && i < it_max
        curr_inds = Determinantal.sample(dpp, n′)
        inds = unique([inds; curr_inds])
        curr_n = Base.size(inds, 1)
        i += 1
    end
    # If the curr. no. of elements is lower than the desired sample size:
    # allow repeated elements
    while curr_n < n
        new_ind = rand(1:curr_n, 1)[1]
        push!(inds, new_ind)
        curr_n += 1
    end
    # If the curr. no. of elements is larger than the desired sample size (n):
    # use the first n elements
    if curr_n > n
        inds = inds[1:n]
    end
    return inds
end

# LSDPP
function create_features(chunk::Matrix)
    return chunk
end
N = size(A, 1)
N′ = ceil(Int, N/2)
lsdpp = LSDPP(Matrix(A); chunksize=min(N′, 4000),
              max=Inf, randomized=false)
function lsdpp_sample(A, n; chunksize=4000, buffersize=1,
                      max=Inf, randomized=false)
    global lsdpp
    inds = sample(lsdpp, n)
    return inds
end

# DPP
# Compute a kernel matrix for the points in x
L = pairwise(Distances.Euclidean(), A')
GC.gc()
# Form an L-ensemble based on the L matrix
dpp = EllEnsemble(L)
function dpp_sample(A, n; distance = Distances.Euclidean())
    global dpp
    # Scale so that the expected size is n
    rescale!(dpp, n)
    # Sample A (obtain indices)
    inds = Determinantal.sample(dpp)
    return inds
end

# Sampling experiments #########################################################

# Define number of experiments
n_experiments = 50

# Define samplers
samplers = [lrdpp_sample, lsdpp_sample, dpp_sample]

# Define batch sample sizes (proportions)
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
    
    # Sampling experiments
    for batch_size_prop in batch_size_props
        for curr_sampler in samplers
            sample_experiment!(res_path, j, curr_sampler, batch_size_prop, n_train, 
                               A, ds_train_rnd, ds_test_rnd, basis, metrics)
            GC.gc()
        end
    end
end

# Postprocess ##################################################################
plotmetrics(res_path, "metrics.csv")

