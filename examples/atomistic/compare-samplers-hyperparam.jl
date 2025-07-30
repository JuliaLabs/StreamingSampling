using PotentialLearning
using OrderedCollections
using AtomsBase, InteratomicPotentials, PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra, Random
using DataFrames, Hyperopt
using Hyperopt: Categorical, Continuous

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
main_path = "/home/eljn/LargeScaleSampling/examples/atomistic/"
res_path  = "$main_path/results-hyperopt-hfo2-balanced-partial-stats/"
run(`mkdir -p $res_path`)

# Load atomistic configurations (random subset of size N)
run(`mkdir -p $res_path`)

ds_path = "$main_path/data/hfox_data_dionysios/"


file_paths  = [ "$ds_path/Hf128_MC_rattled_mp100_form_sorted.extxyz",
                "$ds_path/Hf128_MC_rattled_mp103_form_sorted.extxyz",
                "$ds_path/Hf128_MC_rattled_random_form_sorted.extxyz",
                "$ds_path/Hf2_gas_form_extrapolated.extxyz",
                #"$ds_path/HfB2_MC_rattled2_mp1994_form_sorted.extxyz",
                #"$ds_path/HfB2_MC_rattled_amorphous_form_sorted.extxyz",
                #"$ds_path/HfB2_MC_rattled_mp1994_form_sorted.extxyz",
                #"$ds_path/HfB2_mp1994_EOS_form_sorted.extxyz",
                "$ds_path/Hf_bcc_vacancy_MC_rattled_form_sorted.extxyz",
                "$ds_path/Hf_hcp_vacancy_MC_rattled_form_sorted.extxyz",
                "$ds_path/Hf_MC_rattled_mp1009460_form_sorted.extxyz",
                "$ds_path/Hf_MC_rattled_mp100_form_sorted.extxyz",
                "$ds_path/Hf_MC_rattled_mp103_form_sorted.extxyz",
                "$ds_path/Hf_mp1009460_EOS_convex_hull_form_sorted.extxyz",
                "$ds_path/Hf_mp1009460_EOS_form_sorted.extxyz",
                "$ds_path/Hf_mp100_EOS_form_sorted.extxyz",
                "$ds_path/Hf_mp103_ads_form_sorted.extxyz",
                "$ds_path/Hf_mp103_EOS_convex_hull_form_sorted.extxyz",
                "$ds_path/Hf_mp103_EOS_form_sorted.extxyz",
                "$ds_path/HfO2_bulk_diffusion_barriers_MC_rattled_form_sorted.extxyz",
                #"$ds_path/HfO2_figshare_form_sorted.extxyz",
                "$ds_path/HfO2_MC_rattled2_amorphous_form_sorted.extxyz",
                "$ds_path/HfO2_MC_rattled2_mp1018721_form_sorted.extxyz",
                "$ds_path/HfO2_MC_rattled2_mp352_form_sorted.extxyz",
                "$ds_path/HfO2_MC_rattled2_mp550893_form_sorted.extxyz",
                "$ds_path/HfO2_MC_rattled_mp1018721_form_sorted.extxyz",
                "$ds_path/HfO2_MC_rattled_mp352_form_sorted.extxyz",
                "$ds_path/HfO2_MC_rattled_mp550893_form_sorted.extxyz",
                "$ds_path/HfO2_MC_rattled_random_form_sorted_selected.extxyz",
                "$ds_path/HfO2_mp1018721_EOS_form_sorted.extxyz",
                "$ds_path/HfO2_mp1018721_EOS_xyz_convex_hull_form_sorted.extxyz",
                "$ds_path/HfO2_mp1858_EOS_xyz_convex_hull_form_sorted.extxyz",
                "$ds_path/HfO2_mp352_EOS_convex_hull_form_sorted.extxyz",
                "$ds_path/HfO2_mp352_EOS_form_sorted.extxyz",
                "$ds_path/HfO2_mp352_EOS_press_form_sorted.extxyz",
                "$ds_path/HfO2_mp352_vacancy_MC_rattled_form_sorted.extxyz",
                "$ds_path/HfO2_mp550893_EOS_form_sorted.extxyz",
                "$ds_path/HfO2_mp741_EOS_xyz_convex_hull_form_sorted.extxyz",
                "$ds_path/HfO2_slabs_selected_form_sorted.extxyz",
                "$ds_path/HfO2_stress_mp352_form_sorted.extxyz",
                "$ds_path/HfO_gas_form_extrapolated.extxyz",
                "$ds_path/HfOx_amorphous_MC_rattled_form_sorted.extxyz", # ds4
                "$ds_path/Hf_Ox_hcp_octahedral_MC_rattled_form_sorted.extxyz",
                "$ds_path/Hf_Ox_hcp_octa_tetra_MC_rattled_form_sorted.extxyz",
                "$ds_path/Hf_Ox_hcp_tetrahedral_MC_rattled_form_sorted.extxyz",
                "$ds_path/O2_AL_sel_gen10_form.extxyz",
                "$ds_path/O2_AL_sel_gen11_form.extxyz",
                "$ds_path/O2_AL_sel_gen12_form.extxyz",
                "$ds_path/O2_AL_sel_gen13_form.extxyz",
                "$ds_path/O2_AL_sel_gen14_form.extxyz",
                "$ds_path/O2_AL_sel_gen15_form.extxyz",
                "$ds_path/O2_AL_sel_gen16_form.extxyz",
                "$ds_path/O2_AL_sel_gen17_form.extxyz",
                "$ds_path/O2_AL_sel_gen18_form.extxyz",
                "$ds_path/O2_AL_sel_gen19_form.extxyz",#
                "$ds_path/O2_AL_sel_gen1_form.extxyz",
                "$ds_path/O2_AL_sel_gen2_form.extxyz",
                "$ds_path/O2_AL_sel_gen3_form.extxyz",
                "$ds_path/O2_AL_sel_gen4_form.extxyz",
                "$ds_path/O2_AL_sel_gen5_form.extxyz",
                "$ds_path/O2_AL_sel_gen6_form.extxyz",
                "$ds_path/O2_AL_sel_gen7_form.extxyz",
                "$ds_path/O2_AL_sel_gen8_form.extxyz",
                "$ds_path/O2_AL_sel_gen9_form.extxyz",
                "$ds_path/O2_gas_form_extrapolated.extxyz"]
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

# basis = ACE(species           = [:Hf,:O],
#             body_order        = 4,
#             polynomial_degree = 12,
#             wL                = 2.0,
#             csp               = 1.0,
#             r0                = 1.43,
#             rcutoff           = 4.4 );
# (e_test_mae, e_test_rmse) = (0.13539587460097754, 0.17079928833214678)
# (f_test_mae, f_test_rmse) = (0.16782190053814108, 0.285166842618458)

# Update dataset by adding energy and force descriptors

# println("Computing energy descriptors of dataset...")
# B_time = @elapsed e_descr = compute_local_descriptors(confs, basis)
# println("Computing force descriptors of dataset...")
# dB_time = @elapsed f_descr = compute_force_descriptors(confs, basis)
# GC.gc()
# ds = DataSet(confs .+ e_descr .+ f_descr)

using Serialization
#serialize("hfox_data_dionysios.jls", ds)
ds = deserialize("hfox_data_dionysios.jls")

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
A = stack(sum.(get_values.(get_local_descriptors.(ds_train_rnd))))'

# Samplers #####################################################################

# SRS
function srs_sample(A, n)
    inds = rand(1:size(A, 1), n)
    return inds
end

# LSDPP
function create_features(chunk::Matrix)
    return chunk
end
N = size(A, 1)
N2 = ceil(Int, N/2)
lsdpp = LSDPP(Matrix(A); chunksize=min(N2, 2000),
              max=Inf, randomized=false)
function lsdpp_sample(A, n; chunksize=2000, buffersize=1,
                      max=Inf, randomized=false)
    global lsdpp
    inds = sample(lsdpp, n)
    return inds
end

# DPP
# # Compute a kernel matrix for the points in x
# L = pairwise(Distances.Euclidean(), A')
# GC.gc()
# # Form an L-ensemble based on the L matrix
# dpp = EllEnsemble(L)
# function dpp_sample(A, n; distance = Distances.Euclidean())
#     global dpp
#     # Scale so that the expected size is n
#     rescale!(dpp, n)
#     # Sample A (obtain indices)
#     inds = Determinantal.sample(dpp)
#     return inds
# end

function PotentialLearning.estimate_time(confs, iap; batch_size = 10)
    if length(confs) < batch_size
        batch_size = length(confs)
    end
    random_selector = RandomSelector(length(confs), batch_size)
    inds = PotentialLearning.get_random_subset(random_selector)
    time = @elapsed begin
        f_descr = compute_force_descriptors(confs[inds],
                                            iap.basis,
                                            pbar = false)
        ds = DataSet(confs[inds] .+ f_descr)
        f_pred = get_all_forces(ds, iap)
    end
    n_atoms = sum(length(get_system(c)) for c in confs[inds])
    return time / n_atoms
end

# Sample with SRS and optimize hyperparameters ######################################

batch_size_prop = 0.02
curr_sampler = srs_sample
print("Method:$curr_sampler, batch_size_prop:$batch_size_prop")
exp_path = "$res_path/$curr_sampler-bsp$batch_size_prop/"
run(`mkdir -p $exp_path`)
batch_size = floor(Int, n_train * batch_size_prop)
inds = curr_sampler(A, batch_size)
model = ACE
pars = OrderedDict( :body_order        => [2, 3, 4],
                    :polynomial_degree => [7, 8, 9, 10],
                    :rcutoff           => LinRange(4, 5, 20),
                    :wL                => LinRange(0.5, 2.0, 20),
                    :csp               => LinRange(0.5, 2.0, 20),
                    :r0                => LinRange(0.5, 2.0, 20));
sampler = CLHSampler(dims=[Categorical(3), Categorical(4), Continuous(),
                           Continuous(), Continuous(), Continuous()])
iap, res = hyperlearn!(model, pars, (@views ds_train_rnd[inds]);
                       n_samples = 20, sampler = sampler,
                       ws = [30.0, 1.0], int = true);
GC.gc()
@save_var exp_path iap.β
@save_var exp_path iap.β0
@save_var exp_path iap.basis
@save_dataframe exp_path res
err_time = plot_err_time(res)
@save_fig exp_path err_time
GC.gc()
# Sample with DPP and optimize hyperparameters ######################################
# batch_size_prop = 0.1
# curr_sampler = dpp_sample
# print("Method:$curr_sampler, batch_size_prop:$batch_size_prop")
# exp_path = "$res_path/$curr_sampler-bsp$batch_size_prop/"
# run(`mkdir -p $exp_path`)
# batch_size = floor(Int, n_train * batch_size_prop)
# inds = curr_sampler(A, batch_size)
# model = ACE
# pars = OrderedDict( :body_order        => LinRange(2, 7, 5),
#                     :polynomial_degree => LinRange(7, 12, 5),
#                     :rcutoff           => LinRange(4, 6, 5),
#                     :wL                => LinRange(0.5, 2.0, 5),
#                     :csp               => LinRange(0.5, 2.0, 5),
#                     :r0                => LinRange(0.5, 2.0, 5));
# sampler = CLHSampler(dims=[Categorical(3), Categorical(3), Continuous(),
#                            Continuous(), Continuous(), Continuous()])
# iap, res = hyperlearn!(model, pars, conf_train;
#                        n_samples = 10, sampler = sampler,
#                        loss = custom_loss, ws = [30.0, 1.0], int = true);
# @save_var res_path iap.β
# @save_var res_path iap.β0
# @save_var res_path iap.basis
# @save_dataframe res_path res
# err_time = plot_err_time(res)
# @save_fig res_path err_time


# Sample with LSDPP and optimize hyperparameters ######################################
batch_size_prop = 0.02
curr_sampler = lsdpp_sample
print("Method:$curr_sampler, batch_size_prop:$batch_size_prop")
exp_path = "$res_path/$curr_sampler-bsp$batch_size_prop/"
run(`mkdir -p $exp_path`)
batch_size = floor(Int, n_train * batch_size_prop)
inds = curr_sampler(A, batch_size)
model = ACE
pars = OrderedDict( :body_order        => [2, 3, 4],
                    :polynomial_degree => [7, 8, 9, 10],
                    :rcutoff           => LinRange(4, 5, 20),
                    :wL                => LinRange(0.5, 2.0, 20),
                    :csp               => LinRange(0.5, 2.0, 20),
                    :r0                => LinRange(0.5, 2.0, 20));
sampler = CLHSampler(dims=[Categorical(3), Categorical(4), Continuous(),
                           Continuous(), Continuous(), Continuous()])
iap, res = hyperlearn!(model, pars, (@views ds_train_rnd[inds]);
                       n_samples = 20, sampler = sampler,
                       ws = [30.0, 1.0], int = true);
GC.gc()
@save_var exp_path iap.β
@save_var exp_path iap.β0
@save_var exp_path iap.basis
@save_dataframe exp_path res
err_time = plot_err_time(res)
@save_fig exp_path err_time
GC.gc()