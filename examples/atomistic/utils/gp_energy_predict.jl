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
include("utils/mxp-chol.jl")
# Data #########################################################################

# Define paths and create experiment folder
res_path  = "results-hf-exact/"
res_path  = "results-hf-approx/"
run(`mkdir -p $res_path`)

ds_path= "/home/omairyrm/subsample/DPP"
# Load atomistic configurations (random subset of size N)

T = Float64
metrics = T.(zeros(100, 20))

file_paths = ["$(ds_path)/Hf/config_data/Hf2_gas_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf2_mp103_EOS_1D_form_sorted.extxyz", 
              "$(ds_path)/Hf/config_data/Hf2_mp103_EOS_3D_form_sorted.extxyz", 
              "$(ds_path)/Hf/config_data/Hf2_mp103_EOS_6D_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf128_MC_rattled_mp100_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf128_MC_rattled_mp103_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf128_MC_rattled_random_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf_mp100_EOS_1D_form_sorted.extxyz",
              "$(ds_path)/Hf/config_data/Hf_mp100_primitive_EOS_1D_form_sorted.extxyz"] 
              #"$(ds_path)/HfO2/config_data/HfO2_figshare_form_sorted.extxyz",
              #"$(ds_path)/HfO2/config_data/HfO2_mp352_EOS_1D_form_sorted.extxyz", 
              #"$(ds_path)/HfO2/config_data/HfO2_mp550893_EOS_1D_form_sorted.extxyz",
              #"$(ds_path)/HfO2/config_data/HfO2_mp550893_EOS_6D_form_sorted.extxyz",
              #"$(ds_path)/HfO2/config_data/HfO_EOS_6D_form_sorted.extxyz",
              #"$(ds_path)/HfO2/config_data/HfO_gas_form_sorted.extxyz", 
              #"$(ds_path)/O/config_data/O2_EOS_relax_7D_form_sorted.extxyz", 
              #"$(ds_path)/O/config_data/O2_gas_form_sorted.extxyz", 
              #"$(ds_path)/O/config_data/O2_mp607540_EOS_6D_form_sorted.extxyz", 
              #"$(ds_path)/O/config_data/O_EOS_6D_form_sorted.extxyz"]

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
basis = ACE(species           = [:Hf],
                 body_order        = 6,
                 polynomial_degree = 10,
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

counter=1

for p in [0.2, 0.4, 0.6, 0.8, 1.0]
    for i in [1e1, 1e2, 1e3, 1e4]
        nb=1000
        e_test_metrics_exact,  μpt_exact, σpt_exact =  fit_gpr_exact(res_path, ds_train[1:Int64(floor(n_train*p))], ds_test, basis; gamma=i, lamda=1.0, nb=nb)
        for tol in [1e-4, 1e-3]
            e_test_metrics_approx,  μpt_approx, σpt_approx = fit_gpr_approx(res_path, ds_train[1:Int64(floor(n_train*p))], ds_test, basis, gamma=i, lamda=1.0, tol=tol, nb=nb)        
            μdiff = norm(μpt_exact - μpt_approx) / norm(μpt_exact)
            σdiff =  norm(σpt_approx -  σpt_exact) / norm(σpt_exact)

            metrics[counter, 1] = T(Int64(floor(n_train*p)))
            metrics[counter, 2] = T(length(get_values(get_local_descriptors(ds_train.Configurations[1]))[1]))
            metrics[counter, 3] = T(n_test)
            metrics[counter, 4] = T(p)
            metrics[counter, 5] = T(nb)
            metrics[counter, 6] = T(1.0) #lamda
            metrics[counter, 7] = T(tol)
            metrics[counter, 8] =  T(6) #body order
            metrics[counter, 9] =  T(10) #poly
            metrics[counter, 10] = T(e_test_metrics_exact["e_test_mae"])
            metrics[counter, 11] = T(e_test_metrics_approx["e_test_mae"])
	    metrics[counter, 12] = T(e_test_metrics_exact["e_test_rmse"])
            metrics[counter, 13] = T(e_test_metrics_approx["e_test_rmse"])
            metrics[counter, 14] = T(μdiff)
            metrics[counter, 15] = T(σdiff)
            CSV.write("hf_metrics_$(T)_$(tol)_$(i).csv",  Tables.table(transpose(metrics[counter,:])), writeheader=false, append=true)
            counter += 1
        end
    end
end

#e_test_metrics_appro,  μpt_appro, σpt_appro = fit_gpr_approx(res_path, ds_train, ds_test, basis, gamma=1e1, lamda=1.0, tol=1e-4, nb=1000)

#e_test_metrics_exact,  μpt_appro_exact, σpt_appro_exact =  fit_gpr_exact(res_path, ds_train, ds_test, basis; gamma=1e1, lamda=1.0)

function fit_gpr_exact(path, ds_train, ds_test, basis; gamma=1e1, lamda=1.0, nb=1000)

    lp_train = PotentialLearning.LinearProblem(ds_train)
    lp_test = PotentialLearning.LinearProblem(ds_test)

    B_train = reduce(hcat, lp_train.B)'
    e_train_len = size(B_train, 1)

    B_test = reduce(hcat, lp_test.B)'
    e_test_len = size(B_test, 1)


    train_full = B_train
    train_len = size(train_full, 1)
    test_full = B_test
    test_len = size(test_full, 1)

    dataset_full = [train_full; test_full]

    println("Computing distance matrix")
    time_dist= @elapsed distmat_train = Compute_Distance_Matrix(dataset_full)
    rbf = RBF(ℓ=gamma)
    println("Computing gaussian kernel with ℓ = $gamma")
    time_kernel= @elapsed K_mat= Gaussian_Kernel(distmat_train, rbf)
    K_mat[diagind(K_mat)] .+=lamda #lamda
    GC.gc()    

    Ktt = copy(K_mat[1:train_len, 1:train_len])
    Kk = @view K_mat[1:end, 1:train_len]
    Kpt = @view K_mat[train_len+1:end, 1:train_len]
    Kpp = @view K_mat[train_len+1:end, train_len+1:end]

    e_train = lp_train.e
    ytrue = e_train # we didn't divide by the number of atoms, so we need to do it here????

    e_test = lp_test.e

    BLAS.set_num_threads(1)
    DA_mixed = distribute(Ktt, Blocks(nb, nb))

    region_size = 32
    t_mixed = 0.0

    Dagger.with_options(;scope=Dagger.scope(cuda_gpu=1)) do    
        ScopedValues.with(Dagger.DATADEPS_REGION_SPLIT=>region_size,
        Dagger.DATADEPS_SCHEDULER=>:roundrobin) do
            origin_chol(DA_mixed, LowerTriangular)  
        end
    end

    BLAS.set_num_threads(Threads.nthreads())
    rhs1 = copy(ytrue[:,:])
    Ktt = (collect(DA_mixed))
    BLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ktt, rhs1) 
    BLAS.trsm!('L', 'L', 'T', 'N', 1.0, Ktt, rhs1)

    #rhs1 = copy(ytrue[:,:])
    #time_solve= @elapsed LAPACK.potrf!('L', Ktt) # rhs1 = Ktt \ rhs1 
    #BLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ktt, rhs1) 
    #BLAS.trsm!('L', 'L', 'T', 'N', 1.0, Ktt, rhs1)

    μpt = Kk * rhs1

    μ_train = μpt[1:train_len, :]
    μ_test = μpt[train_len+1:end, :]


    # Get true and predicted values
    n_atoms_train = length.(get_system.(ds_train))
    n_atoms_test = length.(get_system.(ds_test))

    e_train = get_all_energies(ds_train) ./ n_atoms_train
    e_train_pred = μ_train[1:e_train_len, :] ./ n_atoms_train


    e_test = get_all_energies(ds_test) ./ n_atoms_test
    e_test_pred = μ_test[1:e_test_len, :] ./ n_atoms_test


    Ktp = copy(Kpt')
    BLAS.trsm!('L', 'L', 'N', 'N', T(1.0), Ktt, Ktp) 
    BLAS.trsm!('L', 'L', 'T', 'N', T(1.0), Ktt, Ktp)

    σpt = Kpp - Kpt * Ktp

    # Compute metrics
    e_train_metrics = get_metrics(e_train, e_train_pred,
        metrics = [mae, rmse, rsq],
        label = "e_train")

    e_test_metrics = get_metrics(e_test, e_test_pred,
        metrics = [mae, rmse, rsq],
        label = "e_test")

        println("exact:", e_test_metrics)
    

        return e_test_metrics,  μpt, σpt
end

################# Dagger GPU #################


function fit_gpr_approx(path, ds_train, ds_test, basis; gamma=1e1, lamda=1.0, tol=1e-4, nb=1000)

    lp_train = PotentialLearning.LinearProblem(ds_train)
    lp_test = PotentialLearning.LinearProblem(ds_test)

    B_train = reduce(hcat, lp_train.B)'
    e_train_len = size(B_train, 1)

    B_test = reduce(hcat, lp_test.B)'
    e_test_len = size(B_test, 1)


    train_full = B_train
    train_len = size(train_full, 1)
    test_full = B_test
    test_len = size(test_full, 1)

    dataset_full = [train_full; test_full]

    println("Computing distance matrix")
    time_dist= @elapsed distmat_train = Compute_Distance_Matrix(dataset_full)
    rbf = RBF(ℓ=gamma)
    println("Computing gaussian kernel with ℓ = $gamma")
    time_kernel= @elapsed K_mat= Gaussian_Kernel(distmat_train, rbf)
    K_mat[diagind(K_mat)] .+=lamda #lamda
    K_mat_copy = copy(K_mat);
    GC.gc()    


    Ktt = copy(K_mat[1:train_len, 1:train_len])
    Kk = @view K_mat[1:end, 1:train_len]
    Kpt = @view K_mat[train_len+1:end, 1:train_len]
    Kpp = @view K_mat[train_len+1:end, train_len+1:end]

    e_train = lp_train.e
    ytrue = e_train # we didn't divide by the number of atoms, so we need to do it here????

    e_test = lp_test.e

    BLAS.set_num_threads(1)

    DA_mixed = distribute(Ktt, Blocks(nb, nb))
    T = eltype(Ktt)
    println("Adapt precision with tol = $tol")
    MP = adapt_precision(DA_mixed, T(tol))

    region_size = 32
    t_mixed = 0.0

    Dagger.with_options(;scope=Dagger.scope(cuda_gpu=1)) do    
        ScopedValues.with(Dagger.DATADEPS_REGION_SPLIT=>region_size,
        Dagger.DATADEPS_SCHEDULER=>:roundrobin) do
            MixedPrecisionChol!(DA_mixed, LowerTriangular, MP)
        end
    end

    BLAS.set_num_threads(Threads.nthreads())
    rhs1 = copy(ytrue[:,:])
    Ktt = (collect(DA_mixed))
    BLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ktt, rhs1) 
    BLAS.trsm!('L', 'L', 'T', 'N', 1.0, Ktt, rhs1)


    μpt = Kk * rhs1

    μ_train = μpt[1:train_len, :]
    μ_test = μpt[train_len+1:end, :]

    # Get true and predicted values
    n_atoms_train = length.(get_system.(ds_train))
    n_atoms_test = length.(get_system.(ds_test))

    e_train = get_all_energies(ds_train) ./ n_atoms_train
    e_train_pred = μ_train[1:e_train_len, :] ./ n_atoms_train


    e_test = get_all_energies(ds_test) ./ n_atoms_test
    e_test_pred = μ_test[1:e_test_len, :] ./ n_atoms_test

    Ktp = copy(Kpt')
    BLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ktt, Ktp) 
    BLAS.trsm!('L', 'L', 'T', 'N', 1.0, Ktt, Ktp)

    σpt = Kpp - Kpt * Ktp


    # Compute metrics
    e_train_metrics = get_metrics(e_train, e_train_pred,
        metrics = [mae, rmse, rsq],
        label = "e_train")

    e_test_metrics = get_metrics(e_test, e_test_pred,
        metrics = [mae, rmse, rsq],
        label = "e_test")

        println("approx:", e_test_metrics)
    

        return e_test_metrics,  μpt, σpt
end    


#############
###################################
#=
lp_train = PotentialLearning.LinearProblem(ds_train)
lp_test = PotentialLearning.LinearProblem(ds_test)

@views B_train = reduce(hcat, lp_train.B)'
@views B_test = reduce(hcat, lp_test.B)'
@views dB_train = reduce(hcat, lp_train.dB)'
@views e_train = lp_train.e
@views f_train = reduce(vcat, lp_train.f)



#@views b = e_train

#int_col = [ones(size(B_train, 1)); zeros(size(dB_train, 1))]
#@views A = hcat(int_col, [B_train; dB_train])
#@views b = [e_train; f_train]


#@views A = [B_train; dB_train]
#@views b = [e_train; f_train]

#Q = Diagonal([30.0 * ones(length(e_train));
#              1.0 * ones(length(f_train))])

#int_col = ones(size(B_train, 1))
#@views A = hcat(int_col, B_train)
#Q = Diagonal(1.0 * ones(length(e_train)))
#@views b = e_train

@views A = B_train
@views b = e_train
# Calculate coefficients βs.
Q = Diagonal(30.0 * ones(length(e_train)))

βs = Vector{Float64}() 
λ = 0.0
βs = (A'*Q*A + λ*I) \ (A'*Q*b)


# Update lp.
#if int
#lp_train.β0 .= βs[1]
#lp_train.β  .= βs[2:end]
lp_train.β  .= βs

vref_dict=nothing


n_atoms_train = length.(get_system.(ds_train))
n_atoms_test = length.(get_system.(ds_test))

e_train = get_all_energies(ds_train) ./ n_atoms_train
e_train_pred = (B_train * lp_train.β) ./ n_atoms_train

e_test = get_all_energies(ds_test) ./ n_atoms_test
e_test_pred=  (B_test * lp_train.β) ./ n_atoms_test


# Compute metrics
e_train_metrics = get_metrics(e_train, e_train_pred,
                                  metrics = [mae, rmse, rsq],
                                  label = "e_train")


e_test_metrics = get_metrics(e_test, e_test_pred,
                    metrics = [mae, rmse, rsq],
                                 label = "e_test")
=#
