# Functions to plot metrics of sampling comparison experiments

using DataFrames, CSV, Statistics, Plots, StatsBase

function plotmetrics(res_path, metrics_filename)
    metrics = CSV.read("$res_path/$metrics_filename", DataFrame)

    methods = reverse(unique(metrics.method))
    #methods = ["cur_sample", "kmeans_sample", "simple_random_sample"]

    batch_sizes = unique(metrics.batch_size)
    batch_size_prop = unique(metrics.batch_size_prop)
    xticks_label = ("$b\n$(round(p*100, digits=3))%" for (b, p) in zip(batch_sizes, batch_size_prop))
    colors = palette(:tab10)
    metrics_cols = [:e_train_mae, :f_train_mae, :e_test_mae, :f_test_mae, :time]
    metric_labels = ["E MAE | eV/atom",
                     "F MAE | eV/Å",
                     "E MAE | eV/atom",
                     "F MAE | eV/Å",
                     "Time | s"]
    for (i, metric) in enumerate(metrics_cols)
        plot()
        metric_max = 0.0
        for (j, method) in enumerate(methods)
            metric_means = []; metric_se = []
            metric_q3 = []; metric_q2 = []; metric_q1 = []
            for batch_size in batch_sizes
                ms = metrics[ metrics.method .== method .&&
                              metrics.batch_size .== batch_size , metric]

                # Calculation of mean and standard error
                m = mean(ms)
                se = stdm(ms, m) / sqrt(length(ms))
                push!(metric_means, m)
                push!(metric_se, se)

                # Calculation of quantiles
                qs = quantile(ms, [0.25, 0.5, 0.75])
                push!(metric_q3, qs[3])
                push!(metric_q2, qs[2])
                push!(metric_q1, qs[1])
                
                metric_max = maximum([metric_max, maximum(metric_q3)])
            end
            plot!(batch_sizes,
                metric_q2,
                #metric_means,
                #ribbon = metric_se,
                ribbon = (metric_q2 .- metric_q1, metric_q3 .- metric_q2),
                #yerror = (metric_q2 .- metric_q1, metric_q3 .- metric_q2),
                color = colors[j],
                fillalpha=.05,
                label=method)
            plot!(batch_sizes,
                #metric_means,
                metric_q2,
                seriestype = :scatter,
                thickness_scaling = 1.35,
                markersize = 3,
                markerstrokewidth = 0,
                markerstrokecolor = :black, 
                markercolor = colors[j],
                label="")
            #plot!(batch_sizes, [0.1 for _ in 1:length(batch_sizes)]; 
            #      color=:red, linestyle=:dot, label=false)
            max = metric == :time ? 1 : metric_max*1.1 # 1.0
            min = metric == :time ? -0.1 : 0.001 #minimum(metric_q2)*0.5
            plot!(dpi = 300,
                label = "",
                #xscale=:log2,
                #yscale=:log2,
                xticks = (batch_sizes, xticks_label),
                ylim=(min, max),
                xlabel = "Training Dataset Size (Sample Size)",
                ylabel = metric_labels[i])
        end
        plot!(legend=:topright)
        savefig("$res_path/$metric.png")
    end
end
