# Lazy chunk iterator functions: randomized and sequencial implementations

function chunk_iterator(file_paths::Vector{String}; chunksize=100,
                        buffersize=32, randomized=true)
    if randomized
        return chunk_iterator_rnd(file_paths; chunksize=chunksize,
                                  buffersize=buffersize)
    else
        return chunk_iterator_seq(file_paths; chunksize=chunksize,
                                  buffersize=buffersize)
    end
end

function chunk_iterator_rnd(file_paths::Vector{String}; chunksize=100, buffersize=32)
    N = 0
    ch = Channel{Tuple{Vector,Vector{Int}}}(buffersize) do ch
        # First pass to know file positions
        file_numbers = []
        elem_positions = []
        for (i,filepath) in enumerate(file_paths)
            open(filepath, "r") do io
                while !eof(io)
                    push!(file_numbers, i)
                    push!(elem_positions, position(io))
                    element = read_element(io)
                    N += 1
                end
            end
        end
        
        # Randomize
        inds = randperm(length(file_numbers))
        file_numbers = file_numbers[inds]
        elem_positions = elem_positions[inds]

        # Second pass to create randomized chunks
        global_counter = 1
        current_chunk = []
        current_chunk_indices = Int[]
        
        for (file_nbr, elem_pos) in zip(file_numbers, elem_positions)
            open(file_paths[file_nbr], "r") do io
                seek(io, elem_pos)
                element = read_element(io)
                push!(current_chunk, element)
                push!(current_chunk_indices, inds[global_counter])
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
    return ch, N
end

function chunk_iterator_seq(file_paths::Vector{String}; chunksize=100, buffersize=32)
    # First pass to know dataset size
    N = 0
    for (i,filepath) in enumerate(file_paths)
        open(filepath, "r") do io
            while !eof(io)
                read_element(io)
                N += 1
            end
        end
    end
    # Channel definition
    ch = Channel{Tuple{Vector,Vector{Int}}}(buffersize) do ch
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
    return ch, N
end

function chunk_iterator(A::Vector; chunksize=100, buffersize=32, randomized=true)
    N = length(A)
    inds = randomized ? randperm(N) : collect(1:N)
    n_chunks = ceil(Int, N / chunksize)
    ch = Channel{Tuple{Vector,Vector{Int}}}(buffersize) do ch
        for i in 1:n_chunks
            start_idx = (i - 1) * chunksize + 1
            end_idx = min(i * chunksize, N)
            curr_chunk_inds = inds[start_idx:end_idx]
            @views curr_chunk = A[curr_chunk_inds]
            put!(ch, (curr_chunk, curr_chunk_inds))
        end
    end
    return ch, N
end
