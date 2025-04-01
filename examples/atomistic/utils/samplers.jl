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
    L = pairwise(distance, A')
    
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

# LSDPP-based sampling method
function create_features(chunk::Matrix)
    return chunk
end
function lsdpp_sample(A, n; chunksize=5000, buffersize=32,
                      max=Inf, randomized=false)
    N = size(A, 1)
    N′ = ceil(Int, N/2)
    lsdpp = LSDPP(Matrix(A); chunksize=min(N′, chunksize),
                  max=max, randomized=randomized)
    inds = sample(lsdpp, n)
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
    
    # Sample A (obtain indices)
    _, N = Base.size(A)
    n′ = n > N ? N : n
    curr_n = 0
    inds = []
    while curr_n < n
        curr_inds = Determinantal.sample(dpp, n′)
        inds = unique([inds; curr_inds])
        curr_n = Base.size(inds, 1)
    end
    inds = inds[1:n]

    return inds
end
