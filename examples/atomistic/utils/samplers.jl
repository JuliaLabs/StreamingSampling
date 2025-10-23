# Samplers used in sampling comparison experiments

using Clustering
using Determinantal
using LowRankApprox
using DelimitedFiles

# Simple random sampling
function simple_random_sample(A, n)
    n_train = Base.size(A, 1)
    inds = randperm(n_train)[1:n]
    return inds
end

# Kmeans-based sampling method
function kmeans_sample(A, n; k = 5, maxiter=1000)
    c = kmeans(A', k; maxiter=maxiter)
    a = c.assignments # get the assignments of points to clusters
    n_clusters = maximum(a)
    clusters = [findall(x->x==i, a) for i in 1:n_clusters]
    p = n / Base.size(A, 1)
    ns = []
    for c_i in clusters
        n = round(Int64,length(c_i) * p)
        n = n == 0 ? 1 : n
        push!(ns, n)
    end
    inds = reduce(vcat, Clustering.sample.(clusters, ns))
    #inds = reduce(vcat, Clustering.sample.(clusters, [n ÷ length(clusters)]))
    return inds
end

# DPP-based sampling method
function dpp_sample(A, n; distance = Distances.Euclidean())

    # Compute a kernel matrix for the points in x
    D = pairwise(distance, A')
    L = exp.(-D.^2)
    
    # Form an L-ensemble based on the L matrix
    dpp = EllEnsemble(L)
    
    # Scale so that the expected size is n
    rescale!(dpp, n)

    # Sample A (obtain indices)
    inds = Determinantal.sample(dpp)
     
    return inds
end

# CUR-based sampling method
function cur_sample(A, n)
    r, _ = cur(A)
    inds = @views r
    if length(r) > n
        inds = @views r[1:n]
    end
    return inds
end

# Experimental samplers

# StreamMaxEnt-based sampling method
function create_features(chunk::Vector)
    return chunk
end
function create_feature(element::Vector)
    system = element[1]
    feature = sum(compute_local_descriptors(system, basis))
    return feature
end
function sme_sample(A, n; chunksize=4000, buffersize=1,
                      max=Inf, randomized=false)
    N = size(A, 1)
    N′ = ceil(Int, N/2)
    sme = StreamMaxEnt(Matrix(A); chunksize=min(N′, chunksize),
                  max=max, randomized=randomized)
    inds = sample(sme, n)
    return inds
end

# DBSCAN-based sampling method
function dbscan_sample(A, n; min_neighbors = 3,
                              min_cluster_size = 3,
                              metric = Clustering.Euclidean())
    c = dbscan(A', 0.1; min_neighbors = min_neighbors,
                        min_cluster_size = min_cluster_size,
                        metric = metric)
    a = c.assignments # get the assignments of points to clusters
    n_clusters = maximum(a)
    clusters = [findall(x->x==i, a) for i in 1:n_clusters]
    p = n / Base.size(A, 1)
    ns = []
    for c_i in clusters
        n = round(Int64,length(c_i) * p)
        n = n == 0 ? 1 : n
        push!(ns, n)
    end
    inds = reduce(vcat, Clustering.sample.(clusters, ns))
    #inds = reduce(vcat, Clustering.sample.(clusters, [n ÷ length(clusters)]))
    return inds
end

# Low Rank DPP-based sampling method
function lrdpp_sample(A, n)
    # Compute a kernel matrix for the points in x
    L = LowRank(Matrix(A))
    
    # Form an L-ensemble based on the L matrix
    dpp = EllEnsemble(L)
    
    # Sample A (obtain indices). Use resampling if needed.
    _, N = Base.size(A)
    n′ = n > N ? N : n
    curr_n = 0
    inds = []
    it_max = 1000
    i = 0
    while curr_n < n && i < it_max
        curr_inds = Determinantal.sample(dpp, n′)
        inds = unique([inds; curr_inds])
        curr_n = Base.size(inds, 1)
        i += 1
    end

    # If the curr. no. of elements is lower than the desired sample size:
    # allow repeated elements
    while curr_n < n
        new_ind = rand(1:curr_n, 1)[1]
        push!(inds, new_ind)
        curr_n += 1
    end

    # If the curr. no. of elements is larger than the desired sample size (n):
    # use the first n elements
    if curr_n > n
        inds = inds[1:n]
    end

    return inds
end
