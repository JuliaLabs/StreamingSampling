# Functions for performing sampling comparisons

include("./subtract_peratom_e.jl")
include("./gpr_kernel.jl")


function fit_gpr(path, ds_train, ds_test, basis; gamma=1e1, lamda=0.1, uncertainty=false)

    lp_train = PotentialLearning.LinearProblem(ds_train)
    lp_test = PotentialLearning.LinearProblem(ds_test)

    B_train = reduce(hcat, lp_train.B)'
    e_train_len = size(B_train, 1)
    dB_train = reduce(hcat, lp_train.dB)'
    f_train_len = size(dB_train, 1)

    B_test = reduce(hcat, lp_test.B)'
    e_test_len = size(B_test, 1)
    dB_test = reduce(hcat, lp_test.dB)'
    f_test_len = size(dB_test, 1)

    train_full = [B_train; dB_train]
    train_len = size(train_full, 1)
    test_full = [B_test; dB_test]
    test_len = size(test_full, 1)
    
    dataset_full = [train_full; test_full]

    println("Computing distance matrix")
    distmat_train = Compute_Distance_Matrix(dataset_full)
    rbf = RBF(ℓ=1e1)
    println("Computing gaussian kernel with ℓ = $gamma")
    K_mat= Gaussian_Kernel(distmat_train, rbf)
#    K_mat[diagind(K_mat)] .+=1.0 #lamda
    GC.gc()    


    Ktt = copy(K_mat[1:train_len, 1:train_len])
    Ktt[diagind(Ktt)] .+=1.0

    Kk = @view K_mat[1:end, 1:train_len]
    Kpt = @view K_mat[train_len+1:end, 1:train_len]
    Kpp = @view K_mat[train_len+1:end, train_len+1:end]
    
    e_train = lp_train.e
    f_train = reduce(vcat, lp_train.f)
    ytrue = [e_train; f_train] # we didn't divide by the number of atoms, so we need to do it here????
    
    e_test = lp_test.e
    f_test = reduce(vcat, lp_test.f)


    LAPACK.potrf!('L', Ktt)

    rhs1 = copy(ytrue[:,:])
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

    f_train = get_all_forces(ds_train)
    f_train_pred = μ_train[e_train_len+1:f_train_len+e_train_len, :]

    @save_var path e_train
    @save_var path e_train_pred
    @save_var path f_train
    @save_var path f_train_pred


    e_test = get_all_energies(ds_test) ./ n_atoms_test
    #e_test_ind_start=  f_train_len+e_train_len + 1
    #e_test_ind_end=  f_train_len+e_train_len + e_test_len
    e_test_pred = μ_test[1:e_test_len, :] ./ n_atoms_test

    f_test = get_all_forces(ds_test)
    #f_test_ind_start=  f_train_len+e_train_len + e_test_len + 1
    #f_test_ind_end=  f_train_len+e_train_len + e_test_len + f_test_len
    f_test_pred = μ_test[e_test_len+1:e_test_len+f_test_len, :]

    @save_var path e_test
    @save_var path e_test_pred
    @save_var path f_test
    @save_var path f_test_pred
    

    # Compute metrics
    e_train_metrics = get_metrics(e_train, e_train_pred,
        metrics = [mae, rmse, rsq],
        label = "e_train")
    f_train_metrics = get_metrics(f_train, f_train_pred,
        metrics = [mae, rmse, rsq, mean_cos],
        label = "f_train")
    train_metrics = merge(e_train_metrics, f_train_metrics)
    @save_dict path train_metrics

    e_test_metrics = get_metrics(e_test, e_test_pred,
        metrics = [mae, rmse, rsq],
        label = "e_test")
    f_test_metrics = get_metrics(f_test, f_test_pred,
        metrics = [mae, rmse, rsq, mean_cos],
        label = "f_test")
    test_metrics = merge(e_test_metrics, f_test_metrics)
    @save_dict path test_metrics

    # Plot and save results

    e_plot = plot_energy(e_train, e_train_pred,
    e_test, e_test_pred)
    @save_fig path e_plot

    f_plot = plot_forces(f_train, f_train_pred,
    f_test, f_test_pred)
    @save_fig path f_plot

    e_train_plot = plot_energy(e_train, e_train_pred)
    f_train_plot = plot_forces(f_train, f_train_pred)
    f_train_cos  = plot_cos(f_train, f_train_pred)
    @save_fig path e_train_plot
    @save_fig path f_train_plot
    @save_fig path f_train_cos

    e_test_plot = plot_energy(e_test, e_test_pred)
    f_test_plot = plot_forces(f_test, f_test_pred)
    f_test_cos  = plot_cos(f_test, f_test_pred)
    @save_fig path e_test_plot
    @save_fig path f_test_plot
    @save_fig path f_test_cos

    return e_train_metrics, f_train_metrics, 
    e_test_metrics, f_test_metrics


end

# Fit function used to get errors based on sampling
function fit(path, ged_mat, ds_train, ds_test, basis; vref_dict=nothing)
    vref_dict=nothing
    # Learn
    lb = PotentialLearning.LBasisPotential(basis)
    ws, int = [30.0, 1.0], true
    learn!(lb, ds_train, ws, int)
    @save_var path lb.β
    @save_var path lb.β0

    # Post-process output: calculate metrics, create plots, and save results #######

    # Get true and predicted values
    n_atoms_train = length.(get_system.(ds_train))
    n_atoms_test = length.(get_system.(ds_test))
    
    # training energies were permanently modified, so need special function to recover original energies and predicted energies
    if !isnothing(vref_dict)
        e_train = get_all_energies_w_onebody(ds_train,vref_dict) ./n_atoms_train
        e_train_pred = get_all_energies_w_onebody(ds_train,lb,vref_dict) ./n_atoms_train
    else
        e_train, e_train_pred = get_all_energies(ds_train) ./ n_atoms_train,
                                get_all_energies(ds_train, lb) ./ n_atoms_train
    end
    f_train, f_train_pred = get_all_forces(ds_train),
                            get_all_forces(ds_train, lb)
    @save_var path e_train
    @save_var path e_train_pred
    @save_var path f_train
    @save_var path f_train_pred

    # test energies were *not* permanently modified, so only need special function update predicted energies 
    if !isnothing(vref_dict)
        e_test = get_all_energies(ds_test) ./ n_atoms_test
        e_test_pred = get_all_energies_w_onebody(ds_test,lb,vref_dict) ./n_atoms_test
    else
        e_test, e_test_pred = get_all_energies(ds_test) ./ n_atoms_test,
                              get_all_energies(ds_test, lb) ./ n_atoms_test
    end

    f_test, f_test_pred = get_all_forces(ds_test),
                          get_all_forces(ds_test, lb)
    @save_var path e_test
    @save_var path e_test_pred
    @save_var path f_test
    @save_var path f_test_pred

    # Compute metrics
    e_train_metrics = get_metrics(e_train, e_train_pred,
                                  metrics = [mae, rmse, rsq],
                                  label = "e_train")
    f_train_metrics = get_metrics(f_train, f_train_pred,
                                  metrics = [mae, rmse, rsq, mean_cos],
                                  label = "f_train")
    train_metrics = merge(e_train_metrics, f_train_metrics)
    @save_dict path train_metrics

    e_test_metrics = get_metrics(e_test, e_test_pred,
                                 metrics = [mae, rmse, rsq],
                                 label = "e_test")
    f_test_metrics = get_metrics(f_test, f_test_pred,
                                 metrics = [mae, rmse, rsq, mean_cos],
                                 label = "f_test")
    test_metrics = merge(e_test_metrics, f_test_metrics)
    @save_dict path test_metrics

    # Plot and save results

    e_plot = plot_energy(e_train, e_train_pred,
                         e_test, e_test_pred)
    @save_fig path e_plot

    f_plot = plot_forces(f_train, f_train_pred,
                         f_test, f_test_pred)
    @save_fig path f_plot

    e_train_plot = plot_energy(e_train, e_train_pred)
    f_train_plot = plot_forces(f_train, f_train_pred)
    f_train_cos  = plot_cos(f_train, f_train_pred)
    @save_fig path e_train_plot
    @save_fig path f_train_plot
    @save_fig path f_train_cos

    e_test_plot = plot_energy(e_test, e_test_pred)
    f_test_plot = plot_forces(f_test, f_test_pred)
    f_test_cos  = plot_cos(f_test, f_test_pred)
    @save_fig path e_test_plot
    @save_fig path f_test_plot
    @save_fig path f_test_cos
    
    return e_train_metrics, f_train_metrics, 
           e_test_metrics, f_test_metrics
end

# Main sample experiment function
function sample_experiment!(res_path, j, curr_sampler, batch_size_prop, n_train, 
                            ged_mat, ds_train_rnd, ds_test_rnd, basis, metrics;
                            vref_dict=nothing)
    try
        print("Experiment:$j, method:$curr_sampler, batch_size_prop:$batch_size_prop")
        exp_path = "$res_path/$j-$curr_sampler-bsp$batch_size_prop/"
        run(`mkdir -p $exp_path`)
        batch_size = floor(Int, n_train * batch_size_prop)
        sampling_time = @elapsed begin
            inds = curr_sampler(ged_mat, batch_size)
        end
        metrics_j = fit(exp_path, (@views ds_train_rnd[Int64.(inds)]),
                        ds_test_rnd, basis; vref_dict=vref_dict)
        metrics_j = merge(OrderedDict("exp_number" => j,
                                      "method" => "$curr_sampler",
                                      "batch_size_prop" => batch_size_prop,
                                      "batch_size" => batch_size,
                                      "time" => sampling_time),
                    merge(metrics_j...))
        push!(metrics, metrics_j)
        @save_dataframe(res_path, metrics)
        print(", e_test_mae:$(round(metrics_j["e_test_mae"], sigdigits=4)), f_test_mae:$(round(metrics_j["f_test_mae"], sigdigits=4)), time:$(round(sampling_time, sigdigits=4))")
        println()
    catch e # Catch error from excessive matrix allocation.
        println(e)
    end
end
#=
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

    
    DA_mixed = distribute(Ktt, Blocks(nb, nb))
    wait.(DA_mixed.chunks)
    BLAS.set_num_threads(1)
    println("Before starting cholesky")
    region_size = 128
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

    #@save_var path e_train
    #@save_var path e_train_pred

    e_test = get_all_energies(ds_test) ./ n_atoms_test
    e_test_pred = μ_test[1:e_test_len, :] ./ n_atoms_test

    #@save_var path e_test
    #@save_var path e_test_pred

    Ktp = copy(Kpt')
    BLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ktt, Ktp) 
    BLAS.trsm!('L', 'L', 'T', 'N', 1.0, Ktt, Ktp)

    σpt = Kpp - Kpt * Ktp
   #display(σpt)

    # Compute metrics
    e_train_metrics = get_metrics(e_train, e_train_pred,
        metrics = [mae, rmse, rsq],
        label = "e_train")

    e_test_metrics = get_metrics(e_test, e_test_pred,
        metrics = [mae, rmse, rsq],
        label = "e_test")

    #println("exact:", e_test_metrics)
    

    # Plot and save results

    e_plot = plot_energy(e_train, e_train_pred,
    e_test, e_test_pred)
    @save_fig path e_plot

    e_train_plot = plot_energy(e_train, e_train_pred)
    @save_fig path e_train_plot


    e_test_plot = plot_energy(e_test, e_test_pred)
    @save_fig path e_test_plot


        return e_test_metrics,  μpt, σpt
end
=#
################# Dagger GPU #################
#=

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



    DA_mixed = distribute(Ktt, Blocks(nb, nb))
    wait.(DA_mixed.chunks)
    T = eltype(Ktt)
    BLAS.set_num_threads(1)
    println("Adapt precision with tol = $tol")
    MP = adapt_precision(DA_mixed, T(tol))
    #display(MP)
    #Save the matrix to a binary file
    #serialize("$(path)_datatype_matrix_$(tol)_$(gamma)_$(length(ds_train)).dat", MP)

    println("Before starting cholesky")
    region_size = 128
    t_mixed = 0.0
    
    Dagger.with_options(;scope=Dagger.scope(cuda_gpu=1)) do    
        ScopedValues.with(Dagger.DATADEPS_REGION_SPLIT=>region_size,
        Dagger.DATADEPS_SCHEDULER=>:roundrobin) do
            MixedPrecisionChol!(DA_mixed, LowerTriangular, MP)
        end
    end
    
    println("Done Mixed precision with tol = $tol")

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


    #@save_var path e_train
    #@save_var path e_train_pred

    e_test = get_all_energies(ds_test) ./ n_atoms_test
    e_test_pred = μ_test[1:e_test_len, :] ./ n_atoms_test

    #@save_var path e_test
    #@save_var path e_test_pred

    Ktp = copy(Kpt')
    BLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ktt, Ktp) 
    BLAS.trsm!('L', 'L', 'T', 'N', 1.0, Ktt, Ktp)

    σpt = Kpp - Kpt * Ktp
    #display(σpt)


    # Compute metrics
    e_train_metrics = get_metrics(e_train, e_train_pred,
        metrics = [mae, rmse, rsq],
        label = "e_train")

    e_test_metrics = get_metrics(e_test, e_test_pred,
        metrics = [mae, rmse, rsq],
        label = "e_test")

    #println("approx:", e_test_metrics)
    

    # Plot and save results

    e_plot = plot_energy(e_train, e_train_pred,
    e_test, e_test_pred)
    @save_fig path e_plot

    e_train_plot = plot_energy(e_train, e_train_pred)
    @save_fig path e_train_plot


    e_test_plot = plot_energy(e_test, e_test_pred)
    @save_fig path e_test_plot

        return e_test_metrics,  μpt, σpt
end    
=#
#=
function spd_distance(A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64})
    # Convert to full matrix if Symmetric
    A_full = real.(A)
    B_full = real.(B)

    #A_full[diagind(A_full)] .+=10^6
    #B_full[diagind(B_full)] .+=10^6

    ϵ = 1e-10
    A_full += ϵ * I
    B_full += ϵ * I
    
    # Check for SPD property
   # @assert isposdef(A_full) "A must be SPD"
   # @assert isposdef(B_full) "B must be SPD"

    # Generalized eigenvalues of (A, B)
    λ = eigen(A_full, B_full).values
    λ = real.(λ)
    # Distance computation
    return sqrt(sum(log.(λ).^2))
end
=#
#=function spd_distance(A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64})
    # Convert to dense and ensure SPD
    A_full = A + 1000I
    B_full = B + 1000I

    # Generalized eigenvalues (can return complex if not SPD)
    λ = eigen(A_full, B_full).values

    # Use only the real part (imag part is ~1e-25 if matrices are nearly SPD)
    return sqrt(sum(log.(real.(λ)).^2))
end=#
#=
function spd_distance(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    AA = CuArray(A)
    BB = CuArray(B)
    F = cholesky(BB)                 # B = L*Lᵀ

    C = (F.L \ (AA / F.L'))            # C = L⁻¹ * A * L⁻ᵀ
    λ = CUSOLVER.syevd!('N', 'L', C)   # standard symmetric eigvals
    return sqrt(sum(log.(real.(λ)).^2))
end
=#

function spd_distance(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real};
                            ϵ0::Float64=1e-12, growth::Float64=10.0, max_tries::Int=6)
    A = Symmetric( (A + A')/2 )
    B = (B + B')/2


    AA = CuArray(A)
    BB = CuArray(B)
    #F = cholesky(BB)                 # B = L*Lᵀ

    # 2) Try Cholesky with incremental jitter
    F = nothing
    for k in 0:max_tries
        ϵ = ϵ0 * growth^k
        try
            F = cholesky(Symmetric(BB + ϵ*I))
            break
        catch err
            if k == max_tries
                rethrow(err)  # genuinely not PD
            end
        end
    end

    C = (F.L \ (AA / F.L'))            # C = L⁻¹ * A * L⁻ᵀ
    λ = CUSOLVER.syevd!('N', 'L', C)   # standard symmetric eigvals
    return sqrt(sum(log.(real.(λ)).^2))
end



function fit_gpr_exact_recursive(path, ds_train, ds_test, basis; gamma=1e1, lamda=1.0, precisions=[Float64])

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
    dataset_full_g = CuArray(dataset_full)
    println("Computing distance matrix")
    flush(stdout)

    time_dist= @elapsed distmat_train = Compute_Distance_Matrix(dataset_full_g)
    rbf = RBF(ℓ=gamma)
    println("Computing gaussian kernel with ℓ = $gamma")
    flush(stdout)
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

    
    #DA_mixed = distribute(Ktt, Blocks(nb, nb))
    #wait.(DA_mixed.chunks)
    #BLAS.set_num_threads(1)
    println("Before starting exact recursive cholesky")
    flush(stdout)

    
    time_exact = @elapsed CUSOLVER.potrf!('L', Ktt) #potrf_recursive!(Ktt, 4096)
    rhs1 = CuArray(ytrue[:,:])


    tsolve = @elapsed begin 
    CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ktt, rhs1) 
    CUBLAS.trsm!('L', 'L', 'T', 'N', 1.0, Ktt, rhs1)
    end

    μpt = Kk * rhs1

    μ_train = μpt[1:train_len, :]
    μ_test = μpt[train_len+1:end, :]

    # Get true and predicted values
    n_atoms_train = length.(get_system.(ds_train))
    n_atoms_test = length.(get_system.(ds_test))

    e_train = get_all_energies(ds_train) ./ n_atoms_train
    e_train_pred = Array(μ_train[1:e_train_len, :]) ./ n_atoms_train

    #@save_var path e_train
    #@save_var path e_train_pred

    e_test = get_all_energies(ds_test) ./ n_atoms_test
    e_test_pred = Array(μ_test[1:e_test_len, :]) ./ n_atoms_test

    #@save_var path e_test
    #@save_var path e_test_pred

    Ktp = copy(Kpt')

    tsolve += @elapsed begin
    CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ktt, Ktp) 
    CUBLAS.trsm!('L', 'L', 'T', 'N', 1.0, Ktt, Ktp)
    end


    σpt = Kpp - Kpt * Ktp
    σpt = Array(σpt)
    #σpt = Kpp - Kpt * Ktp
   #display(σpt)

    # Compute metrics
    e_train_metrics = get_metrics(e_train, e_train_pred,
        metrics = [mae, rmse, rsq],
        label = "e_train")

    e_test_metrics = get_metrics(e_test, e_test_pred,
        metrics = [mae, rmse, rsq],
        label = "e_test")

    #println("exact:", e_test_metrics)
    

    # Plot and save results
#=
    e_plot = plot_energy(e_train, e_train_pred,
    e_test, e_test_pred)
    @save_fig path e_plot

    e_train_plot = plot_energy(e_train, e_train_pred)
    @save_fig path e_train_plot


    e_test_plot = plot_energy(e_test, e_test_pred)
    @save_fig path e_test_plot
=#
        return e_test_metrics,  μpt, σpt, time_exact, tsolve
end

function fit_gpr_approx_recursive(path, ds_train, ds_test, basis; gamma=1e1, lamda=1.0, tol=1e-4, precisions=[Float16, Float16, Float16, Float16, Float16, Float64])

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
    dataset_full_g = CuArray(dataset_full)
    dataset_full = nothing
    println("Computing distance matrix")
    flush(stdout)
    time_dist= @elapsed distmat_train = Compute_Distance_Matrix(dataset_full_g)
    rbf = RBF(ℓ=gamma)
    println("Computing gaussian kernel with ℓ = $gamma")
    flush(stdout)
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

    println("Before starting mixed precision recursive cholesky")
    flush(stdout)

    Ktt_to_factor = SymmMixedPrec(Ktt, 'L'; precisions=precisions);
    time_approx = @elapsed potrf_recursive!(Ktt_to_factor);
    Ktt_to_factor = reconstruct_matrix(Ktt_to_factor)

    rhs1 = CuArray(ytrue[:,:])

    tsolve = @elapsed begin
    CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ktt_to_factor, rhs1) 
    CUBLAS.trsm!('L', 'L', 'T', 'N', 1.0, Ktt_to_factor, rhs1)
    end

    μpt = Kk * rhs1

    μ_train = μpt[1:train_len, :]
    μ_test = μpt[train_len+1:end, :]

    # Get true and predicted values
    n_atoms_train = length.(get_system.(ds_train))
    n_atoms_test = length.(get_system.(ds_test))

    e_train = get_all_energies(ds_train) ./ n_atoms_train
    e_train_pred = Array(μ_train[1:e_train_len, :]) ./ n_atoms_train

    #@save_var path e_train
    #@save_var path e_train_pred

    e_test = get_all_energies(ds_test) ./ n_atoms_test
    e_test_pred = Array(μ_test[1:e_test_len, :]) ./ n_atoms_test

    #@save_var path e_test
    #@save_var path e_test_pred

    Ktp = copy(Kpt')
    
    tsolve += @elapsed begin
    CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0, Ktt_to_factor, Ktp) 
    CUBLAS.trsm!('L', 'L', 'T', 'N', 1.0, Ktt_to_factor, Ktp)
    end 


    σpt = Array(Kpp - Kpt * Ktp)
    #σpt = Kpp - Kpt * Ktp
    #display(σpt)


    # Compute metrics
    e_train_metrics = get_metrics(e_train, e_train_pred,
        metrics = [mae, rmse, rsq],
        label = "e_train")

    e_test_metrics = get_metrics(e_test, e_test_pred,
        metrics = [mae, rmse, rsq],
        label = "e_test")

    #println("approx:", e_test_metrics)
    

    # Plot and save results
#=
    e_plot = plot_energy(e_train, e_train_pred,
    e_test, e_test_pred)
    @save_fig path e_plot

    e_train_plot = plot_energy(e_train, e_train_pred)
    @save_fig path e_train_plot


    e_test_plot = plot_energy(e_test, e_test_pred)
    @save_fig path e_test_plot
=#
        return e_test_metrics,  μpt, σpt, time_approx, tsolve
end    
