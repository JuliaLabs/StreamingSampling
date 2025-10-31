#using Pkg
#Pkg.develop(path="../../")

using StreamingSampling

include("utils/utils.jl")

# Define paths and create experiment folder
train_path = ["data/rmd17/aspirin-train.xyz"]
test_path = ["data/rmd17/aspirin-test.xyz"]
res_path  = "results-aspirin-rmd17/"
run(`mkdir -p $res_path`)

# Initialize streaming sampling ################################################
read_element(io) = read_element_xyz(io)
basis = ACE(species           = [:C, :O, :H],
            body_order        = 4,
            polynomial_degree = 8,
            wL                = 2.0,
            csp               = 1.0,
            r0                = 1.43,
            rcutoff           = 4.4 );
function create_feature(element::Vector; basis=basis)
    system = element[1]
    feature = sum(compute_local_descriptors(system, basis))
    return feature
end
ws = compute_weights(train_path;
                     read_element=read_element,
                     create_feature=create_feature,
                     chunksize=2000,
                     subchunksize=200)
open("ws-aspirin-rmd17.jls", "w") do io
    serialize(io, ws)
    flush(io)
end
#ws = deserialize("ws-aspirin-rmd17.jls")

# Sampling experiments #########################################################

# Define number of experiments
n_experiments = 1

# Define batch sample sizes
sample_sizes = [1_000, 5_000, 10_000]

# Test dataset size
m = 10_000

# Full dataset size
N = length(ws)

# Define basis for fitting
basis_fitting = ACE(species           = [:C, :O, :H],
                    body_order        = 4,
                    polynomial_degree = 12,
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
ch, _ = chunk_iterator(train_path;
                       read_element=read_element,
                       chunksize=n1,
                       buffersize=1,
                       randomized=true)
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
    ch, _ = chunk_iterator(test_path;
                           read_element=read_element,
                           chunksize=m,
                           buffersize=1,
                           randomized=true)
    cs, test_inds = take!(ch)
    close(ch)
    test_confs = []
    for c in cs
        system, energy, forces = c
        conf = Configuration(system, Energy(energy),
                             Forces([Force(f) for f in forces]))
        push!(test_confs, conf)
    end
    ds_test = DataSet(test_confs)
    ds_test = calc_descr!(ds_test, basis_fitting)
    open("test-ds-aspirin-rmd17.jls", "w") do io
        serialize(io, ds_test)
        flush(io)
    end
    #ds_test = deserialize("test-ds-aspirin-rmd17.jls")
    
    for n in sample_sizes
        # Sample training dataset using streaming weighted sampling ############
        train_inds = StatsBase.sample(1:length(ws), Weights(ws), n;
                     replace=false, ordered=true)
        #Load atomistic configurations
        ds_train = get_confs(train_path, read_element, train_inds)
        #Adjust reference energies (permanent change)
        adjust_energies!(ds_train, vref_dict)
        # Compute dataset with energy and force descriptors
        ds_train = calc_descr!(ds_train, basis_fitting)
        # Create result folder
        curr_sampler = "sws"
        exp_path = "$res_path/$j-$curr_sampler-n$n/"
        run(`mkdir -p $exp_path`)
        # Fit and save results
        metrics_j = fit(exp_path, ds_train, ds_test, basis_fitting; vref_dict=vref_dict)
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
        ds_train = get_confs(train_path, read_element, train_inds)
        #Adjust reference energies (permanent change)
        adjust_energies!(ds_train, vref_dict)
        # Compute dataset with energy and force descriptors
        ds_train = calc_descr!(ds_train, basis_fitting)
        # Create result folder
        curr_sampler = "srs"
        exp_path = "$res_path/$j-$curr_sampler-n$n/"
        run(`mkdir -p $exp_path`)
        # Fit and save results
        metrics_j = fit(exp_path, ds_train, ds_test, basis_fitting; vref_dict=vref_dict)
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
plot_err_per_sample(res_path, "metrics.csv")

