# Generic feature calculation functions

function read_element(io::IO)
    return readline(io)
end

function create_feature(element::String)
    return parse.(Float64, split(element))
end

function create_features(chunk::AbstractVector{T};
                         create_feature=create_feature) where T
    chunksize = length(chunk)
    features = Vector{Any}(undef, chunksize)
    Threads.@threads for i in eachindex(chunk)
        features[i] = create_feature(chunk[i])
    end
    return Matrix(hcat(features...)')
end

