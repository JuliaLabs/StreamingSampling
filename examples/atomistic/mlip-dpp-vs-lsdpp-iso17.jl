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
res_path  = "results-full-iso17/"
run(`mkdir -p $res_path`)

# Helper functions
function get_confs(path, inds)
    confs = []
    ch, N = chunk_iterator(train_path; chunksize=1000, randomized=false)
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

# Sampling experiments #########################################################

# Setup LSDPP
basis = ACE(species           = [:C, :O, :H],
            body_order        = 4,
            polynomial_degree = 6,
            wL                = 2.0,
            csp               = 1.0,
            r0                = 1.43,
            rcutoff           = 4.4 );
lsdpp = LSDPP(train_path; chunksize=2000, subchunksize=200)
open("lsdpp.jls", "w") do io
    serialize(io, lsdpp)
    flush(io)
end
#lsdpp = deserialize("lsdpp.jls")

# Define number of experiments
n_experiments = 1

# Define batch sample sizes
sample_sizes = [1_000, 5_000, 10_000]

# Test dataset size
m = 10_000

# Full dataset size
N = length(lsdpp.weights)

# Define basis for fitting
basis_fitting = ACE(species           = [:C, :O, :H],
                    body_order        = 4,
                    polynomial_degree = 16,
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

# Compute reference energies
s = 0.0
n1 = 10_000
ch, _ = chunk_iterator(train_path; chunksize=n1, buffersize=1, randomized=true)
c, _ = take!(ch)
close(ch)
for cj in c
    global s
    energy = cj[2]
    s += energy
end
na = length(c[1][1]) # all conf. have the same no. of atoms
avg_energy_per_atom = s/n1/na
vref_dict = Dict(:H => avg_energy_per_atom,
                 :C => avg_energy_per_atom,
                 :O => avg_energy_per_atom)

# Run experiments
for j in 1:n_experiments
    println("Experiment $j")

    global metrics
    local ch 

    # Create test set
    ch, _ = chunk_iterator(test_path; chunksize=m, buffersize=1, randomized=true)
    _, test_inds = take!(ch)
    close(ch)
    test_inds = sort(test_inds)
    test_confs = get_confs(test_path, test_inds)
    test_ds = calc_descr(test_confs, basis_fitting)
    open("test_ds.jls", "w") do io
     serialize(io, test_ds)
     flush(io)
    end
    #test_ds = deserialize("test_ds_.jls")
    
    for n in sample_sizes
        # Sample training dataset using LSDPP ##################################
        train_inds = sort(sample(lsdpp, n))
        #Load atomistic configurations
        train_confs = get_confs(train_path, train_inds)
        #Adjust reference energies (permanent change)
        adjust_energies(train_confs,vref_dict)
        # Compute dataset with energy and force descriptors
        train_ds = calc_descr(train_confs, basis_fitting)
        # Create result folder
        curr_sampler = "lsdpp"
        exp_path = "$res_path/$j-$curr_sampler-n$n/"
        run(`mkdir -p $exp_path`)
        # Fit and save results
        metrics_j = fit(exp_path, train_ds, test_ds, basis_fitting; vref_dict=vref_dict)
        metrics_j = merge(OrderedDict("exp_number" => j,
                                      "method" => "$curr_sampler",
                                      "batch_size_prop" => n/N,
                                      "batch_size" => n,
                                      "time" => 0.0),
                    merge(metrics_j...))
        push!(metrics, metrics_j)
        @save_dataframe(res_path, metrics)
        GC.gc()
        
        # Sample training dataset using SRS ####################################
        train_inds = randperm(N)[1:n]
        
        #Load atomistic configurations
        train_confs = get_confs(train_path, train_inds)
        # Compute dataset with energy and force descriptors
        train_ds = calc_descr(train_confs, basis_fitting)
        # Create result folder
        curr_sampler = "srs"
        exp_path = "$res_path/$j-$curr_sampler-n$n/"
        run(`mkdir -p $exp_path`)
        # Fit and save results
        metrics_j = fit(exp_path, train_ds, test_ds, basis_fitting; vref_dict=vref_dict)
        metrics_j = merge(OrderedDict("exp_number" => j,
                                      "method" => "$curr_sampler",
                                      "batch_size_prop" => n/N,
                                      "batch_size" => n,
                                      "time" => 0.0),
                    merge(metrics_j...))
        push!(metrics, metrics_j)
        @save_dataframe(res_path, metrics)
        GC.gc()
    end
end

# Postprocess ##################################################################
plotmetrics2(res_path, "metrics.csv")

