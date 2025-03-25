# Lazy chunk iterator functions: randomized and sequencial implementations

function chunk_iterator(file_paths::Vector{String}; chunksize=200, randomized=true)
    if randomized
        return chunk_iterator_rnd(file_paths; chunksize=chunksize)
    else
        return chunk_iterator_seq(file_paths; chunksize=chunksize)
    end
end

function chunk_iterator_rnd(file_paths::Vector{String}; chunksize=200)
    return Channel{Tuple{Vector,Vector{Int}}}(32) do ch
        # First pass to know file positions
        file_numbers = []
        file_positions = []
        for (i,filepath) in enumerate(file_paths)
            open(filepath, "r") do io
                while !eof(io)
                    push!(file_numbers, i)
                    push!(file_positions, position(io))
                    element = read_element(io)
                end
            end
        end
        
        # Randomize
        inds = sortperm(file_numbers)
        file_numbers = file_numbers[inds]
        file_positions = file_positions[inds]
        
        # Second pass to create randomized chunks
        global_counter = 1
        current_chunk = []
        current_chunk_indices = Int[]
        
        for (file_nbr, file_pos) in zip(file_numbers, file_positions)
            open(file_paths[file_nbr], "r") do io
                seek(io, file_pos)
                element = read_element(io)
                push!(current_chunk, element)
                push!(current_chunk_indices, global_counter)
                global_counter += 1
                if length(current_chunk) == chunksize
                    put!(ch, (copy(current_chunk), copy(current_chunk_indices)))
                    empty!(current_chunk)
                    empty!(current_chunk_indices)
                end
            end
        end
        if !isempty(current_chunk)
            put!(ch, (current_chunk, current_chunk_indices))
        end
    end
end

function chunk_iterator_seq(file_paths::Vector{String}; chunksize=200)
    return Channel{Tuple{Vector,Vector{Int}}}(32) do ch
        global_counter = 1
        current_chunk = []
        current_chunk_indices = Int[]
        for file in file_paths
            open(file, "r") do io
                while !eof(io)
                    element = read_element(io)
                    push!(current_chunk, element)
                    push!(current_chunk_indices, global_counter)
                    global_counter += 1
                    if length(current_chunk) == chunksize
                        put!(ch, (copy(current_chunk), copy(current_chunk_indices)))
                        empty!(current_chunk)
                        empty!(current_chunk_indices)
                    end
                end
            end
        end
        if !isempty(current_chunk)
            put!(ch, (current_chunk, current_chunk_indices))
        end
    end
end

