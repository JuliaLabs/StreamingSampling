# helper
padlims(v; frac=0.10) = begin
    vmin, vmax = extrema(v)
    span = max(vmax - vmin, eps())
    (vmin - frac*span, vmax + frac*span)
end

function plot_weights(ws, inds;
        size=(640, 480), dpi=300,
        guidefs=20, tickfs=14, legendfs=16,
        grid=:y,
        margins=(left=30, right=40, bottom=30, top=20))

    perm = sortperm(ws)
    s    = ws[perm]
    x    = collect(eachindex(s))
    m    = in(Set(inds)).(perm)

    # ensure margins are AbsoluteLength in px (avoids Int + AbsoluteLength conflicts)
    lp = margins.left  * Plots.px
    rp = margins.right * Plots.px
    bp = margins.bottom* Plots.px
    tp = margins.top   * Plots.px

    plt = plot(size=size, dpi=dpi, grid=grid, framestyle=:box,
               left_margin=lp, right_margin=rp,
               bottom_margin=bp, top_margin=tp,
               xguidefont=font(guidefs), yguidefont=font(guidefs),
               xtickfont=font(tickfs),  ytickfont=font(tickfs),
               legendfont=font(legendfs))

    scatter!(plt, x, s;
        color=:gray, alpha=0.25,
        marker=:circle, markersize=8, markerstrokewidth=0,
        label="All",
        xlabel="Sorted element indices",
        ylabel="Weights",
        ylims=padlims(s))

    if any(m)
        scatter!(plt, x[m], s[m];
            color=:red, marker=:utriangle,
            markersize=10, markerstrokewidth=0,
            label="Selected")
    else
        plot!(plt, NaN, NaN; label="Selected")
    end

    return plt
end

