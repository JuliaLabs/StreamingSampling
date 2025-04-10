# Functions for performing sampling comparisons

include("./subtract_peratom_e.jl")

# Fit function used to get errors based on sampling
function fit(path, ds_train, ds_test, basis; vref_dict=nothing)

    # Learn
    lb = PotentialLearning.LBasisPotential(basis)
    # no intercept , out-of-core formulation of linear learn
    ws = [30.0, 1.0]
    _AtWA, _AtWb = PotentialLearning.ooc_learn!(lb, ds_train;ws=ws,symmetrize=false, λ=0.01)
    @save_var path lb.β

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

