# Synthetic data generation

function generate_data(file_paths::Vector{String}; feature_size=50)
    # Define an feature_size-dimensional mean vector
    mu = zeros(feature_size)
    # Define an feature_size x feature_size covariance matrix
    Sigma = Matrix(I, feature_size, feature_size)
    # Create the feature_size-dimensional multivariate normal distribution
    mvnorm = MvNormal(mu, Sigma)
    points = Vector{Float64}[]
    # Create synthetic files
    for (i, path) in enumerate(file_paths)
        open(path, "w") do io
            for j in 1:1000
                p = rand(mvnorm)
                if rand() < 0.04
                    p .+= rand(-4:4, feature_size)
                end
                p_str = reduce(*, "$(po) " for po in p)
                push!(points, p)
                println(io, "$(p_str)")
            end
        end
    end
    return points
end

