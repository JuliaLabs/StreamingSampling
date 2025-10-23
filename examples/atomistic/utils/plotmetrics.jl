using DataFrames, CSV, Statistics, Plots, StatsBase,Printf, Measures

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

function plotmetrics2(res_path::String)
    # ---------------- Load & prep ----------------
    df  = CSV.read("$res_path/metrics.csv", DataFrame)
    sort!(df, [:batch_size])

    srs = filter(:method => ==("srs"),   df)
    spd = filter(:method => ==("lsdpp"), df)   # labeled as SPD

    # ---------------- Percent formatting (round UP, fixed) ----------------
    # ≥ 1%  -> ceil to integer (no decimals)
    # < 1%  -> ceil to one decimal; drop .0 if it’s effectively an integer
    format_percent_roundup(p::Float64) = begin
        perc = 100 * p
        if perc >= 1
            return string(Int(ceil(perc)), "%")
        else
            val = ceil(perc * 10) / 10        # e.g., 0.99 -> 1.0
            if isapprox(val, round(val); atol=1e-12)
                return string(Int(round(val)), "%")
            else
                return @sprintf("%.1f%%", val)
            end
        end
    end

    # ---------------- X tick labels ----------------
    xs = spd.batch_size
    xtick_labels = [string(bs, "\n", format_percent_roundup(prop))
                    for (bs, prop) in zip(spd.batch_size, spd.batch_size_prop)]

    # ---------------- Colors ----------------
    black = RGB(0,0,0)
    red   = RGB(0.75,0.10,0.10)

    # ---------------- Helpers ----------------
    padlims(v; frac=0.10) = begin
        vmin, vmax = minimum(v), maximum(v)
        span = max(vmax - vmin, eps())
        (vmin - frac*span, vmax + frac*span)
    end

    # ---------------- Global styling ----------------
    default(
        fontfamily        = "Computer Modern",
        linewidth         = 5.5,
        markersize        = 12,
        markerstrokewidth = 1.8,
        guidefont         = font(26),     # axis labels
        tickfont          = font(24),     # tick labels
        legendfont        = font(26),     # legend
        dpi               = 600,
        size              = (1100, 1100), # large square figure
        grid              = :y,
        framestyle        = :box,
        left_margin       = 8mm,
        right_margin      = 14mm,         # avoid clipping trailing '%'
        bottom_margin     = 10mm,
        top_margin        = 6mm,
    )

    # ======================= ENERGY =======================
    pE_top = plot(
        srs.batch_size, srs.e_test_mae;
        color = black, lw = 5.5, marker = :circle,
        xlabel = "", ylabel = "E MAE | eV/atom",
        label = "SRS",
        xticks = (xs, xtick_labels),
        legend = :topright,
    )

    pE_bottom = plot(
        spd.batch_size, spd.e_test_mae;
        color = red, lw = 5.5, marker = :utriangle,
        xlabel = "Training Dataset Size (Sample Size)",
        ylabel = "E MAE | eV/atom",
        label = "SPD",
        xticks = (xs, xtick_labels),
        legend = :topright,
        ylims = padlims(spd.e_test_mae),
    )

    energy_plot = plot(pE_top, pE_bottom; layout=(2,1), size=(1100,1100))
    savefig(energy_plot, "$res_path/e_test_mae_by_sample.pdf")

    # ======================= FORCE =======================
    pF_top = plot(
        srs.batch_size, srs.f_test_mae;
        color = black, lw = 5.5, marker = :circle,
        xlabel = "", ylabel = "F MAE | eV/Å",
        label = "SRS",
        xticks = (xs, xtick_labels),
        legend = :topright,
    )

    pF_bottom = plot(
        spd.batch_size, spd.f_test_mae;
        color = red, lw = 5.5, marker = :utriangle,
        xlabel = "Training Dataset Size (Sample Size)",
        ylabel = "F MAE | eV/Å",
        label = "SPD",
        xticks = (xs, xtick_labels),
        legend = :topright,
        ylims = padlims(spd.f_test_mae),
    )

    force_plot = plot(pF_top, pF_bottom; layout=(2,1), size=(1100,1100))
    savefig(force_plot, "$res_path/f_test_mae_by_sample.pdf")

    println("✅ Saved:")
    println(" - e_test_mae_by_sample.pdf")
    println(" - f_test_mae_by_sample.pdf")
end
