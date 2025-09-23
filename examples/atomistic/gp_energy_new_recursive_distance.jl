using PotentialLearning
using OrderedCollections
using Serialization
using KernelAbstractions
include("../../../NextLA.jl/src/cholesky_tree_subdiv.jl")
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
#include("utils/mxp-chol.jl")
# Data #########################################################################

function execut()
# Define paths and create experiment folder
res_path_exact  = "results-hfando-exact/"
res_path_approx  = "results-hfando-approx/"
run(`mkdir -p $res_path_exact`)
run(`mkdir -p $res_path_approx`)

ds_path= "/home/rabab53/data/"
# Load atomistic configurations (random subset of size N)

T = Float64
metrics = T.(zeros(1000, 24))

file_paths = ["$(ds_path)Hf2_gas_form_sorted.extxyz",
              "$(ds_path)Hf2_mp103_EOS_1D_form_sorted.extxyz", 
              "$(ds_path)Hf2_mp103_EOS_3D_form_sorted.extxyz",
	      "$(ds_path)Hf2_mp103_EOS_6D_form_sorted.extxyz",
              "$(ds_path)Hf128_MC_rattled_mp100_form_sorted.extxyz",
              "$(ds_path)Hf128_MC_rattled_mp103_form_sorted.extxyz",
              "$(ds_path)Hf128_MC_rattled_random_form_sorted.extxyz",
              "$(ds_path)Hf_mp100_EOS_1D_form_sorted.extxyz",
              "$(ds_path)Hf_mp100_primitive_EOS_1D_form_sorted.extxyz",
              "$(ds_path)HfO2_figshare_form_sorted.extxyz",
              "$(ds_path)HfO2_mp352_EOS_1D_form_sorted.extxyz", 
              "$(ds_path)HfO2_mp550893_EOS_1D_form_sorted.extxyz",
              "$(ds_path)HfO2_mp550893_EOS_6D_form_sorted.extxyz",
              "$(ds_path)HfO_EOS_6D_form_sorted.extxyz",
              "$(ds_path)HfO_gas_form_sorted.extxyz"
              ] 

#=
file_paths = [
"$(ds_path)/HfO2/config_data/HfO2_figshare_form_sorted.extxyz",
"$(ds_path)/HfO2/config_data/HfO2_mp352_EOS_1D_form_sorted.extxyz", 
"$(ds_path)/HfO2/config_data/HfO2_mp550893_EOS_1D_form_sorted.extxyz",
"$(ds_path)/HfO2/config_data/HfO2_mp550893_EOS_6D_form_sorted.extxyz",
"$(ds_path)/HfO2/config_data/HfO_EOS_6D_form_sorted.extxyz",
"$(ds_path)/HfO2/config_data/HfO_gas_form_sorted.extxyz"]
=#
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
basis = ACE(species           = [:Hf, :O],
                 body_order        = 5,
                 polynomial_degree = 8,
                 wL                = 2.0,
                 csp               = 1.0,
                 r0                = 1.43,
                 rcutoff           = 6.0 );

# Update dataset by adding energy and force descriptors
println("Computing energy descriptors of dataset...")
B_time = @elapsed e_descr = compute_local_descriptors(confs, basis)
println("Computing force descriptors of dataset...")
dB_time = @elapsed f_descr = compute_force_descriptors(confs, basis)
#GC.gc()
ds = DataSet(confs .+ e_descr .+ f_descr)

for j in 1:25
# Define randomized training and test dataset.
# Here, both datasets have elements of each file.
rnd_inds_train = Int[]
rnd_inds_test = Int[]
for (i, ni) in enumerate(confsizes)
    n_train_i = floor(Int, 0.86 * ni)
    n_test_i = ni - n_train_i
    rnd_inds = randperm(ni) .+ offsets[i]
    push!(rnd_inds_train, rnd_inds[1:n_train_i]...)
    push!(rnd_inds_test, rnd_inds[n_train_i+1:n_train_i+n_test_i]...)
end
ds_train = @views ds[rnd_inds_train]
ds_test  = @views ds[rnd_inds_test]
n_train = length(ds_train)
n_test = length(ds_test)
println("n_train = $n_train, n_test = $n_test")
count=1
n_test=10240

      for p in [10240, 20480, 30720, 40960, 51200, 61440] #[0.2, 0.4, 0.6, 0.8, 1.0]
        for i in [1e1] #, 1e2, 1e3, 1e4]
        
        rnd_inds = randperm(n_train)
        #ds_train_copy = deepcopy(ds_train)
        #ds_train_copy =  @views ds_train_copy[rnd_inds]
        ds_train = @views ds_train[rnd_inds]
        spath_exact = "$res_path_exact/$p-$i/"
        run(`mkdir -p $spath_exact`)
        nb=2000
       tt1= @elapsed e_test_metrics_exact,  μpt_exact, σpt_exact, time_exact, time_solve_exact =  fit_gpr_exact_recursive(spath_exact, ds_train[1:p], ds_test[1:n_test], basis; gamma=i, lamda=1000.0, precisions=[Float64])
        for tol in [1e-4]#, 1e-6]
            spath_approx = "$res_path_approx/$p-$i-$tol/"
            run(`mkdir -p $spath_approx`)

	    for ti in [[Float16, Float16, Float16, Float16, Float16, Float16, Float32, Float64], [Float16, Float16, Float16, Float16, Float16, Float32, Float64], [Float16, Float16, Float16, Float16, Float32, Float64], [Float16, Float16, Float16, Float32, Float64], [Float16, Float16, Float32, Float64], [Float16, Float32, Float64]]

	    tt2=@elapsed e_test_metrics_approx,  μpt_approx, σpt_approx, time_approx, time_solve_approx = fit_gpr_approx_recursive(spath_approx, ds_train[1:p], ds_test[1:n_test], basis; gamma=i, lamda=1000.0, tol=tol, precisions=ti)        
            μdiff = norm(μpt_exact - μpt_approx) / norm(μpt_exact)
            σdiff =  norm(σpt_approx -  σpt_exact) / norm(σpt_exact)
            μdiff_abs = norm(μpt_exact - μpt_approx)
            σdiff_abs =  norm(σpt_approx -  σpt_exact)
            metrics[count, 1] = p #T(Int64(floor(n_train*p)))
            metrics[count, 2] = T(length(get_values(get_local_descriptors(ds_train.Configurations[1]))[1]))
            metrics[count, 3] = T(n_test)
            metrics[count, 4] = T(0.0)
            metrics[count, 5] = T(0.0)
            metrics[count, 6] = T(1000.0) #lamda
            metrics[count, 7] = length(ti)
            metrics[count, 8] =  T(5) #body order
            metrics[count, 9] =  T(8) #poly
            metrics[count, 10] = T(e_test_metrics_exact["e_test_mae"]/2.0)
            metrics[count, 11] = T(e_test_metrics_approx["e_test_mae"]/2.0)
            metrics[count, 12] = T(e_test_metrics_exact["e_test_rmse"]/2.0)
            metrics[count, 13] = T(e_test_metrics_approx["e_test_rmse"]/2.0)
            metrics[count, 14] = T(μdiff)
            metrics[count, 15] = T(σdiff)
            metrics[count, 16] = T(μdiff_abs)
            metrics[count, 17] = T(σdiff_abs)
            #println("Start spd_distance")
            metrics[count, 18] = spd_distance(σpt_exact, σpt_approx)
            metrics[count, 19] = time_exact
            metrics[count, 20] = time_approx
            metrics[count, 21] = time_solve_exact
            metrics[count, 22] = time_solve_approx
	    metrics[count, 23] = time_exact + time_solve_exact
            metrics[count, 24] = time_approx + time_solve_approx

            CSV.write("recursive_hfo_metrics_$(i)_$(p)_add_distance.csv",  Tables.table(transpose(metrics[count,:])), writeheader=false, append=true)
            #println("end spd_distance")
            println(transpose(metrics[count,:]))
            println((tt1, tt2))
            flush(stdout)
	    count += 1
	    end
        end
    end
end
end
CSV.write("all_together.csv",  Tables.table(transpose(metrics[:,:])), writeheader=false, append=true)

#e_test_metrics_appro,  μpt_appro, σpt_appro = fit_gpr_approx(res_path, ds_train, ds_test, basis, gamma=1e1, lamda=1.0, tol=1e-4, nb=1000)

#e_test_metrics_exact,  μpt_appro_exact, σpt_appro_exact =  fit_gpr_exact(res_path, ds_train, ds_test, basis; gamma=1e1, lamda=1.0)
end

execut()
