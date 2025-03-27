# Lazy chunk iterator functions: randomized and sequencial implementations

function chunk_iterator(file_paths::Vector{String}; chunksize=200,
                        buffersize=32, randomized=true)
    if randomized
        return chunk_iterator_rnd(file_paths; chunksize=chunksize,
                                  buffersize=buffersize)
    else
        return chunk_iterator_seq(file_paths; chunksize=chunksize,
                                  buffersize=buffersize)
    end
end

function chunk_iterator_rnd(file_paths::Vector{String}; chunksize=200, buffersize=32)
    return Channel{Tuple{Vector,Vector{Int}}}(buffersize) do ch
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
        inds = randperm(length(file_numbers))
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

function chunk_iterator_seq(file_paths::Vector{String}; chunksize=200, buffersize=32)
    return Channel{Tuple{Vector,Vector{Int}}}(buffersize) do ch
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

function chunk_iterator(A::Matrix; chunksize=200, buffersize=32, randomized=true)
    N = size(A, 1)
    inds = randomized ? randperm(N) : collect(1:N)
    n_chunks = ceil(Int, N / chunksize)
    return Channel{Tuple{Matrix,Vector{Int}}}(buffersize) do ch
        for i in 1:n_chunks
            start_idx = (i - 1) * chunksize + 1
            end_idx = min(i * chunksize, N)
            curr_chunk_inds = inds[start_idx:end_idx]
            @views curr_chunk = A[curr_chunk_inds, :]
            put!(ch, (curr_chunk, curr_chunk_inds))
        end
    end
end
