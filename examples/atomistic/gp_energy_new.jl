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
include("utils/mxp-chol.jl")
# Data #########################################################################

function execut()
# Define paths and create experiment folder
res_path_exact  = "results-hfando-exact/"
res_path_approx  = "results-hfando-approx/"
run(`mkdir -p $res_path_exact`)
run(`mkdir -p $res_path_approx`)

ds_path= "/home/omairyrm/subsample/DPP"
# Load atomistic configurations (random subset of size N)

T = Float64
metrics = T.(zeros(1000, 20))

file_paths = ["$(ds_path)/Hf/config_data/Hf2_gas_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf2_mp103_EOS_1D_form_sorted.extxyz", 
              "$(ds_path)/Hf/config_data/Hf2_mp103_EOS_3D_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf2_mp103_EOS_6D_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf128_MC_rattled_mp100_form_sorted.extxyz"
              "$(ds_path)/Hf/config_data/Hf128_MC_rattled_mp103_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf128_MC_rattled_random_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf_mp100_EOS_1D_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf_mp100_primitive_EOS_1D_form_sorted.extxyz",
              "$(ds_path)/HfO2/config_data/HfO2_figshare_form_sorted.extxyz",
              "$(ds_path)/HfO2/config_data/HfO2_mp352_EOS_1D_form_sorted.extxyz", 
              "$(ds_path)/HfO2/config_data/HfO2_mp550893_EOS_1D_form_sorted.extxyz",
              "$(ds_path)/HfO2/config_data/HfO2_mp550893_EOS_6D_form_sorted.extxyz",
              "$(ds_path)/HfO2/config_data/HfO_EOS_6D_form_sorted.extxyz",
              "$(ds_path)/HfO2/config_data/HfO_gas_form_sorted.extxyz"] 

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
ds_train = @views ds[rnd_inds_train]
ds_test  = @views ds[rnd_inds_test]
n_train = length(ds_train)
n_test = length(ds_test)
println("n_train = $n_train, n_test = $n_test")
count=1

for p in [0.2, 0.4, 0.6, 0.8, 1.0]
    for i in [1e1] #[1e1, 1e2, 1e3, 1e4]
        for j in 1:50
        rnd_inds = randperm(n_train)
        ds_train_copy = deepcopy(ds_train)
        ds_train_copy =  @views ds_train_copy[rnd_inds]
        spath_exact = "$res_path_exact/$p-$i/"
        run(`mkdir -p $spath_exact`)
        nb=1000
        e_test_metrics_exact,  μpt_exact, σpt_exact =  fit_gpr_exact(spath_exact, ds_train_copy[1:Int64(floor(n_train*p))], ds_test, basis; gamma=i, lamda=1.0, nb=nb)
        for tol in [1e-4]#, 1e-6]
            spath_approx = "$res_path_approx/$p-$i-$tol/"
            run(`mkdir -p $spath_approx`)
            e_test_metrics_approx,  μpt_approx, σpt_approx = fit_gpr_approx(spath_approx, ds_train_copy[1:Int64(floor(n_train*p))], ds_test, basis, gamma=i, lamda=1.0, tol=tol, nb=nb)        
            μdiff = norm(μpt_exact - μpt_approx) / norm(μpt_exact)
            σdiff =  norm(σpt_approx -  σpt_exact) / norm(σpt_exact)
            μdiff_abs = norm(μpt_exact - μpt_approx)
            σdiff_abs =  norm(σpt_approx -  σpt_exact)
            metrics[count, 1] = T(Int64(floor(n_train*p)))
            metrics[count, 2] = T(length(get_values(get_local_descriptors(ds_train.Configurations[1]))[1]))
            metrics[count, 3] = T(n_test)
            metrics[count, 4] = T(p)
            metrics[count, 5] = T(nb)
            metrics[count, 6] = T(1.0) #lamda
            metrics[count, 7] = T(tol)
            metrics[count, 8] =  T(5) #body order
            metrics[count, 9] =  T(8) #poly
            metrics[count, 10] = T(e_test_metrics_exact["e_test_mae"])
            metrics[count, 11] = T(e_test_metrics_approx["e_test_mae"])
            metrics[count, 12] = T(e_test_metrics_exact["e_test_rmse"])
            metrics[count, 13] = T(e_test_metrics_approx["e_test_rmse"])
            metrics[count, 14] = T(μdiff)
            metrics[count, 15] = T(σdiff)
            metrics[count, 16] = T(μdiff_abs)
            metrics[count, 17] = T(σdiff_abs)
            metrics[count, 18] = spd_distance(σpt_exact, σpt_approx)
            CSV.write("hfo_metrics_$(T)_$(tol)_$(i)_$(p).csv",  Tables.table(transpose(metrics[count,:])), writeheader=false, append=true)
            println(transpose(metrics[count,:]))
            count += 1
        end
    end
    end
end
CSV.write("all_together.csv",  Tables.table(transpose(metrics[:,:])), writeheader=false, append=true)

#e_test_metrics_appro,  μpt_appro, σpt_appro = fit_gpr_approx(res_path, ds_train, ds_test, basis, gamma=1e1, lamda=1.0, tol=1e-4, nb=1000)

#e_test_metrics_exact,  μpt_appro_exact, σpt_appro_exact =  fit_gpr_exact(res_path, ds_train, ds_test, basis; gamma=1e1, lamda=1.0)
end

execut()