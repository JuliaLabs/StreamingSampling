using CSV, DataFrames, Plots, Printf, Measures

# ---------------- Load & prep ----------------
df  = CSV.read("metrics.csv", DataFrame)
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
savefig(energy_plot, "e_test_mae_by_sample_iso17.pdf")

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
savefig(force_plot, "f_test_mae_by_sample_iso17.pdf")

println("✅ Saved:")
println(" - e_test_mae_by_sample_iso17.pdf")
println(" - f_test_mae_by_sample_iso17.pdf")

