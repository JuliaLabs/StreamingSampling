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
main_path = "/home/eljn/LargeScaleSampling/examples/atomistic/"
res_path  = "$main_path/results-hfo2-balanced-partial-stats/"
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

basis = ACE(species           = [:Hf,:O],
            body_order        = 4,
            polynomial_degree = 12,
            wL                = 2.0,
            csp               = 1.0,
            r0                = 1.43,
            rcutoff           = 4.4 );
# (e_test_mae, e_test_rmse) = (0.13539587460097754, 0.17079928833214678)
# (f_test_mae, f_test_rmse) = (0.16782190053814108, 0.285166842618458)

# Update dataset by adding energy and force descriptors

println("Computing energy descriptors of dataset...")
#B_time = @elapsed e_descr = compute_local_descriptors(confs, basis)
#println("Computing force descriptors of dataset...")
#dB_time = @elapsed f_descr = compute_force_descriptors(confs, basis)
#GC.gc()
#ds = DataSet(confs .+ e_descr .+ f_descr)

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
#serialize("rnd_inds_train.jls", rnd_inds_train)
#serialize("rnd_inds_test.jls", rnd_inds_test)
rnd_inds_train = deserialize("rnd_inds_train.jls")
rnd_inds_test = deserialize("rnd_inds_test.jls")

ds_train_rnd = @views ds[rnd_inds_train]
ds_test_rnd  = @views ds[rnd_inds_test]
n_train = length(ds_train_rnd)
n_test = length(ds_test_rnd)
A = stack(sum.(get_values.(get_local_descriptors.(ds_train_rnd))))'



# Samplers #####################################################################


# LSDPP
function create_features(chunk::Matrix)
    return chunk
end
N = size(A, 1)
N2 = ceil(Int, N/2)
lsdpp = LSDPP(Matrix(A); chunksize=min(N2, 1000), subchunksize=100,
              max=Inf, randomized=false)
function lsdpp_sample(A, n; chunksize=1000, subchunksize=100, buffersize=1,
                      max=Inf, randomized=false)
    global lsdpp
    inds = sample(lsdpp, n)
    return inds
end

# DPP
# Compute a kernel matrix for the points in x
#L = pairwise(Distances.Euclidean(), A')

#serialize("L.jls", L)
L = deserialize("L.jls")

# Form an L-ensemble based on the L matrix
dpp = EllEnsemble(L)

serialize("dpp.jls", dpp)

L = nothing; GC.gc()
function dpp_sample(A, n; distance = Distances.Euclidean())
    global dpp
    # Scale so that the expected size is n
    rescale!(dpp, n)
    # Sample A (obtain indices)
    inds = Determinantal.sample(dpp)
    return inds
end

# # Sampling experiments #########################################################

# # Define number of experiments
# n_experiments = 40

# # Define samplers
# samplers = [dpp_sample, lsdpp_sample] #, lrdpp_sample]

# # Define batch sample sizes (proportions)
# batch_size_props = [0.08, 0.16, 0.32, 0.64] #[0.08, 0.16, 0.32, 0.64]

# # Create metric dataframe
# metric_names = [:exp_number,  :method, :batch_size_prop, :batch_size, :time,
#                 :e_train_mae, :e_train_rmse, :e_train_rsq,
#                 :f_train_mae, :f_train_rmse, :f_train_rsq, :f_train_mean_cos,
#                 :e_test_mae,  :e_test_rmse,  :e_test_rsq, 
#                 :f_test_mae,  :f_test_rmse,  :f_test_rsq,  :f_test_mean_cos]
# metrics = DataFrame([Any[] for _ in 1:length(metric_names)], metric_names)

# # Run experiments
# for j in 1:n_experiments
#     global metrics
    
#     # Sampling experiments
#     for batch_size_prop in batch_size_props
#         for curr_sampler in samplers
#             sample_experiment!(res_path, j, curr_sampler, batch_size_prop, n_train, 
#                               A, ds_train_rnd, ds_test_rnd, basis, metrics)
#             GC.gc()
#         end
#     end
# end

# # Postprocess ##################################################################
# plotmetrics(res_path, "metrics.csv")
