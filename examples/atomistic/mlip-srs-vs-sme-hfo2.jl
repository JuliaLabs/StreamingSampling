using PotentialLearning
using OrderedCollections
using Serialization

# LSSampling
include("../../src/StreamingSampling.jl")

# Domain specific functions
include("utils/macros.jl")
include("utils/aux_sample_functions.jl")
include("utils/plots.jl")
include("utils/plotmetrics.jl")
include("utils/atom-conf-features-extxyz.jl")
include("utils/xyz.jl")

# Define paths and create experiment folder
ds_paths = ["Hf_mp1009460_EOS_form_sorted.extxyz",
            "HfO2_mp352_EOS_press_form_sorted.extxyz",
            "O2_AL_sel_gen13_form.extxyz",
            "HfO2_mp741_EOS_xyz_convex_hull_form_sorted.extxyz",
            "HfO2_MC_rattled2_amorphous_form_sorted.extxyz",
            "Hf2_gas_form_extrapolated.extxyz",
            "Hf_MC_rattled_mp103_form_sorted.extxyz",
            "HfO2_figshare_form_sorted.extxyz",
            "HfO_gas_form_extrapolated.extxyz",
            "Hf_MC_rattled_mp100_form_sorted.extxyz",
            "Hf_mp1009460_EOS_convex_hull_form_sorted.extxyz",
            "HfO2_MC_rattled_mp1018721_form_sorted.extxyz",
            "Hf_mp100_EOS_form_sorted.extxyz",
            "O2_AL_sel_gen6_form.extxyz",
            "HfO2_stress_mp352_form_sorted.extxyz",
            "O2_AL_sel_gen14_form.extxyz",
            "Hf128_MC_rattled_mp103_form_sorted.extxyz",
            "Hf_mp103_ads_form_sorted.extxyz",
            #"HfB2_MC_rattled_mp1994_form_sorted.extxyz",
            "O2_gas_form_extrapolated.extxyz",
            "Hf_Ox_hcp_tetrahedral_MC_rattled_form_sorted.extxyz",
            "O2_AL_sel_gen10_form.extxyz",
            "HfO2_mp352_EOS_convex_hull_form_sorted.extxyz",
            "O2_AL_sel_gen15_form.extxyz",
            "O2_AL_sel_gen9_form.extxyz",
            "HfO2_MC_rattled2_mp1018721_form_sorted.extxyz",
            "HfO2_MC_rattled_random_form_sorted_selected.extxyz",
            "O2_AL_sel_gen12_form.extxyz",
            "O2_AL_sel_gen8_form.extxyz",
            "HfO2_mp1858_EOS_xyz_convex_hull_form_sorted.extxyz",
            #"HfB2_MC_rattled_amorphous_form_sorted.extxyz",
            "O2subsampling/O2_A2_sel_cur.extxyz",
            "O2_AL_sel_gen18_form.extxyz",
            "HfO2_slabs_selected_form_sorted.extxyz",
            "O2_AL_sel_gen4_form.extxyz",
            "O2_AL_sel_gen2_form.extxyz",
            "HfO2_mp1018721_EOS_xyz_convex_hull_form_sorted.extxyz",
            "Hf128_MC_rattled_random_form_sorted.extxyz",
            "HfO2_MC_rattled_mp550893_form_sorted.extxyz",
            #"HfB2_MC_rattled2_mp1994_form_sorted.extxyz",
            "Hf_mp103_EOS_form_sorted.extxyz",
            "O2_AL_sel_gen3_form.extxyz",
            "HfO2_MC_rattled2_mp352_form_sorted.extxyz",
            #"HfB2_mp1994_EOS_form_sorted.extxyz",
            "O2_AL_sel_gen11_form.extxyz",
            "HfO2_bulk_diffusion_barriers_MC_rattled_form_sorted.extxyz",
            "HfO2_mp352_EOS_form_sorted.extxyz",
            "HfO2_MC_rattled_mp352_form_sorted.extxyz",
            "O2_AL_sel_gen1_form.extxyz",
            "HfOx_amorphous_MC_rattled_form_sorted.extxyz",
            "HfO2_mp352_vacancy_MC_rattled_form_sorted.extxyz",
            "Hf_hcp_vacancy_MC_rattled_form_sorted.extxyz",
            "HfO2_mp550893_EOS_form_sorted.extxyz",
            "O2_AL_sel_gen7_form.extxyz",
            "Hf_Ox_hcp_octahedral_MC_rattled_form_sorted.extxyz",
            "O2_AL_sel_gen16_form.extxyz",
            "O2_AL_sel_gen19_form.extxyz",
            "Hf_mp103_EOS_convex_hull_form_sorted.extxyz",
            "Hf_bcc_vacancy_MC_rattled_form_sorted.extxyz",
            "Hf_Ox_hcp_octa_tetra_MC_rattled_form_sorted.extxyz",
            "Hf128_MC_rattled_mp100_form_sorted.extxyz",
            "HfO2_MC_rattled2_mp550893_form_sorted.extxyz",
            "Hf_MC_rattled_mp1009460_form_sorted.extxyz",
            "HfO2_mp1018721_EOS_form_sorted.extxyz",
            "O2_AL_sel_gen17_form.extxyz",
            "O2_AL_sel_gen5_form.extxyz"]
ds_paths = "data/hfox_data_spencer/" .* ds_paths # N=30712

res_path  = "results-hfo2/"
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

function calc_descr!(confs, basis)
    println("Computing energy descriptors of dataset...")
    e_descr = compute_local_descriptors(confs, basis)
    println("Computing force descriptors of dataset...")
    f_descr = compute_force_descriptors(confs, basis)
    GC.gc()
    confs = DataSet(confs .+ e_descr .+ f_descr)
end

function create_feature(element::Configuration)
    system = get_system(element)
    feature = sum(compute_local_descriptors(system, basis))
    return feature
end

# Sampling experiments #########################################################

# Define randomized training and test dataset.
# Here, both datasets have elements of each file.
confs = []
confsizes = zeros(Int, length(ds_paths))
for (i, ds_path) in enumerate(ds_paths)
    newconfs = load_data(ds_path, uparse("eV"), uparse("Å"))
    push!(confs, newconfs...)
    confsizes[i] = length(newconfs)
end
offsets = zeros(Int, length(ds_paths))
for i in 2:length(ds_paths)
    offsets[i] = confsizes[i-1] + offsets[i-1]
end
ds = DataSet(confs)
N = length(confs)
GC.gc()

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

# Compute reference energies
s = 0.0
m = 100#10_000
ch, _ = chunk_iterator(ds_paths; chunksize=m, buffersize=1, randomized=true)
c, _ = take!(ch)
close(ch)
for cj in c
    global s
    energy = cj[2]
    s += energy
end
na = length(c[1][1]) # all conf. have the same no. of atoms
avg_energy_per_atom = s/m/na
vref_dict = Dict(:Hf => avg_energy_per_atom,
                 :O => avg_energy_per_atom)

#Adjust reference energies (permanent change)
adjust_energies(ds_train_rnd,vref_dict)

# Define basis for fitting
basis_fitting = ACE(species           = [:Hf, :O],
                    body_order        = 2,
                    polynomial_degree = 3,
                    wL                = 2.0,
                    csp               = 1.0,
                    r0                = 1.43,
                    rcutoff           = 4.4 );
calc_descr!(ds_train_rnd, basis_fitting)
calc_descr!(ds_test_rnd, basis_fitting)

# Setup StreamMaxEnt
basis = ACE(species           = [:Hf, :O],
            body_order        = 2,
            polynomial_degree = 3,
            wL                = 2.0,
            csp               = 1.0,
            r0                = 1.43,
            rcutoff           = 4.4 );
sme = StreamMaxEnt(ds_train_rnd.Configurations; chunksize=2000, subchunksize=200)
open("sme-h2o2.jls", "w") do io
    serialize(io, sme)
    flush(io)
end
#sme = deserialize("sme-h2o2.jls")

# Define number of experiments
n_experiments = 1

# Define batch sample sizes
sample_sizes = [1_000, 5_000, 10_000]

# Full dataset size
N = length(sme.weights)

# Create metric dataframe
metric_names = [:exp_number,  :method, :batch_size_prop, :batch_size, :time,
                :e_train_mae, :e_train_rmse, :e_train_rsq,
                :f_train_mae, :f_train_rmse, :f_train_rsq, :f_train_mean_cos,
                :e_test_mae,  :e_test_rmse,  :e_test_rsq, 
                :f_test_mae,  :f_test_rmse,  :f_test_rsq,  :f_test_mean_cos]
metrics = DataFrame([Any[] for _ in 1:length(metric_names)], metric_names)

# Run experiments
for j in 1:n_experiments
    println("Experiment $j")

    global metrics

    for n in sample_sizes
        # Sample training dataset using StreamMaxEnt ##################################
        train_inds = sort(sample(sme, n))
        #Load atomistic configurations
        train_ds = @views ds_train_rnd[train_inds]
        # Create result folder
        curr_sampler = "sme"
        exp_path = "$res_path/$j-$curr_sampler-n$n/"
        run(`mkdir -p $exp_path`)
        # Fit and save results
        metrics_j = fit(exp_path, train_ds, ds_test_rnd, basis_fitting; vref_dict=vref_dict)
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
        train_ds = @views ds_train_rnd[train_inds]
        # Create result folder
        curr_sampler = "srs"
        exp_path = "$res_path/$j-$curr_sampler-n$n/"
        run(`mkdir -p $exp_path`)
        # Fit and save results
        metrics_j = fit(exp_path, train_ds, ds_test_rnd, basis_fitting; vref_dict=vref_dict)
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

