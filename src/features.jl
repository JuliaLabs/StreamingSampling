# Generic feature calculation functions

function read_element(io::IO)
    return readline(io)
end

function create_feature(element)
    return parse.(Float64, split(element))
end

function create_features(chunk::AbstractVector{T}) where T
    chunksize = length(chunk)
    features = Vector{Any}(undef, chunksize)
    Threads.@threads for i in eachindex(chunk)
        features[i] = create_feature(chunk[i])
    end
    return Matrix(hcat(features...)')
end

