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
include("utils/subtract_peratom_e.jl")


# plot

plotmetrics("results/results-hf-balanced-partial-stats-metrics", "results-hf-balanced-partial-stats-metrics.csv")
plotmetrics("results/results-hf-balanced-metrics", "results-hf-balanced-metrics.csv")



# Plot
plotmetrics("results/results-dpps-hf-balanced-partial-stats-metrics", "results-dpps-hf-balanced-partial-stats-metrics.csv")
plotmetrics("results/results-dpps-iso17-partial-stats-metrics", "results-dpps-iso17-partial-stats-metrics.csv")
plotmetrics("results/results-hf-balanced-40-metrics", "results-hf-balanced-40-metrics.csv")
plotmetrics("results/results-iso17-40-metrics", "results-iso17-40-metrics.csv")
plotmetrics("results/results-iso17-metrics-recentpush", "results-iso17-metrics-recentpush.csv")
plotmetrics("results/results-iso17-partial-stats-metrics", "results-iso17-partial-stats-metrics.csv")


#plotmetrics("results/res-hf-partial", "metrics.csv")
#plotmetrics("res-hf", "metrics.csv")
##plotmetrics("results-iso17", "metrics.csv")


#using CSV
#using DataFrames

## Read the CSV file
#metrics_iso = CSV.read("preliminary-results-iso17/metrics.csv", DataFrame)

#dpp_e =metrics_iso[metrics_iso.method .== "dpp_sample", :][:, :e_test_mae]
#lsdpp_e =metrics_iso[metrics_iso.method .== "lsdpp_sample", :][:, :e_test_mae]
#scatter(dpp_e, lsdpp_e)
#scatter!(xlims=(0, 50), ylims=(0, 50))

#plot(dpp_e)
#plot!(lsdpp_e)
#plot!(ylims=(0, 1))


#dpp_f =metrics_iso[metrics_iso.method .== "dpp_sample", :][:, :f_test_mae]
#lsdpp_f =metrics_iso[metrics_iso.method .== "lsdpp_sample", :][:, :f_test_mae]
#scatter(dpp_f, lsdpp_f)
#scatter!(xlims=(0, 50), ylims=(0, 50))





