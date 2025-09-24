using PotentialLearning
using OrderedCollections
using Serialization

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

# Define paths and create experiment folder
train_path = ["data/iso17/my_iso17_train.extxyz"]
test_path = ["data/iso17/my_iso17_test.extxyz"]
res_path  = "results-full-hfo2/"
run(`mkdir -p $res_path`)

# Helper functions
function get_confs(path, inds)
    confs = []
    ch = chunk_iterator(path; chunksize=1000, randomized=false)
    k = 1
    for (c, ci) in ch
        j = 1
        while j <= length(c) && k <= length(inds)
            if inds[k] == ci[j]
                system, energy, forces = c[j]
                conf = Configuration(system, Energy(energy),
                                     Forces([Force(f) for f in forces]))
                push!(confs, conf)
                k += 1
            end
            j += 1
        end
        if k > length(inds)
            break
        end
    end
    return DataSet(confs)
end

function calc_descr(confs, basis)
    println("Computing energy descriptors of dataset...")
    e_descr = compute_local_descriptors(confs, basis)
    println("Computing force descriptors of dataset...")
    f_descr = compute_force_descriptors(confs, basis)
    GC.gc()
    ds = DataSet(confs .+ e_descr .+ f_descr)
    return ds
end

# Define basis
basis = ACE(species           = [:C, :O, :H],
            body_order        = 4,
            polynomial_degree = 4, #16,
            wL                = 2.0,
            csp               = 1.0,
            r0                = 1.43,
            rcutoff           = 4.4 );

# Create metric dataframe
metric_names = [:exp_number,  :method, :batch_size_prop, :batch_size, :time,
                :e_train_mae, :e_train_rmse, :e_train_rsq,
                :f_train_mae, :f_train_rmse, :f_train_rsq, :f_train_mean_cos,
                :e_test_mae,  :e_test_rmse,  :e_test_rsq, 
                :f_test_mae,  :f_test_rmse,  :f_test_rsq,  :f_test_mean_cos]
metrics = DataFrame([Any[] for _ in 1:length(metric_names)], metric_names)


# Sampling experiments #########################################################

# Define number of experiments
n_experiments = 20

# Define batch sample sizes
sample_sizes = [1000]

# Setup LSDPP
lsdpp = deserialize("lsdpp.jls") # lsdpp = LSDPP(train_path; chunksize=2000, subchunksize=200)


# Run experiments
for j in 1:n_experiments
    println("Experiment $j")

    global metrics

    # Create test set
    ch = chunk_iterator(test_path; chunksize=1000, buffersize=1, randomized=true)
    _, test_inds = take!(ch)
    close(ch)
    test_inds = sort(test_inds)
    test_confs = get_confs(test_path, test_inds)
    test_ds = calc_descr(test_confs, basis)
    
    for n in sample_sizes
        # Sample training dataset using LSDPP ##################################
        train_inds = sort(sample(lsdpp, n))
        #Load atomistic configurations
        train_confs = get_confs(train_path, train_inds)
        # Compute dataset with energy and force descriptors
        train_ds = calc_descr(train_confs, basis)
        # Create result folder
        curr_sampler = "lsdpp"
        exp_path = "$res_path/$j-$curr_sampler-n$n/"
        run(`mkdir -p $exp_path`)
        # Fit and save results
        metrics_j = fit(exp_path, train_ds, test_ds, basis; vref_dict=nothing)
        metrics_j = merge(OrderedDict("exp_number" => j,
                                      "method" => "$curr_sampler",
                                      "batch_size_prop" => 0.0,
                                      "batch_size" => n,
                                      "time" => 0.0),
                    merge(metrics_j...))
        push!(metrics, metrics_j)
        @save_dataframe(res_path, metrics)
        GC.gc()
        
        # Sample training dataset using SRS ####################################
        ch = chunk_iterator(test_path; chunksize=n, buffersize=1, randomized=true)
        _, train_inds = take!(ch)
        close(ch)
        train_inds = sort(train_inds)
        
        #Load atomistic configurations
        train_confs = get_confs(train_path, train_inds)
        # Compute dataset with energy and force descriptors
        train_ds = calc_descr(train_confs, basis)
        # Create result folder
        curr_sampler = "srs"
        exp_path = "$res_path/$j-$curr_sampler-n$n/"
        run(`mkdir -p $exp_path`)
        # Fit and save results
        metrics_j = fit(exp_path, train_ds, test_ds, basis; vref_dict=nothing)
        metrics_j = merge(OrderedDict("exp_number" => j,
                                      "method" => "$curr_sampler",
                                      "batch_size_prop" => 0.0,
                                      "batch_size" => n,
                                      "time" => 0.0),
                    merge(metrics_j...))
        push!(metrics, metrics_j)
        @save_dataframe(res_path, metrics)
        GC.gc()
    end
end

# Postprocess ##################################################################
plotmetrics(res_path, "metrics.csv")

