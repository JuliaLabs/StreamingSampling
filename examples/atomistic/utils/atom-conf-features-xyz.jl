using Unitful
using AtomsBase
using InteratomicPotentials
using StaticArrays

# Domain specific feature calculation

# Create a feature vector of an atomistic system using ACE
function create_feature(element::Vector)
    system = element[1]
    feature = sum(compute_local_descriptors(system, basis))
    return feature
end

# Reads an atomistic system: (XYZ files)
function read_element(io::IO)

    energy_units = uparse("eV")
    distance_units = uparse("Å")
    T = Float64

    system = nothing
    energy = nothing
    forces = []
    
    if !eof(io)
    
        # Read number of atoms
        line = readline(io)
        num_atoms = parse(Int, line)
        
         # Read energy
        line = readline(io)
        energy = parse(T, line)

        # Read atoms
        atoms = Vector{AtomsBase.Atom}(undef, num_atoms)
        for i = 1:num_atoms
            line = split(readline(io))
            line_count = 1
            data = Dict(())
            element = Symbol(line[line_count])
            line_count += 1
            position = SVector{3}(parse.(T, line[line_count:line_count+2]))
            line_count += 3
            push!(forces, parse.(T, line[line_count:line_count+2]))
            line_count += 3
            if isempty(data)
                atoms[i] = AtomsBase.Atom(element, position .* distance_units)
            else
                atoms[i] =
                    AtomsBase.Atom(element, position .* distance_units, data...)
            end
        end

        xmin = minimum(minimum.(position.(atoms)))
        xmax = maximum(maximum.(position.(atoms)))
        ε = abs(xmax - xmin) / 100
        xmin -= ε; xmax += ε
        box = [[xmax, xmin, xmin],
               [xmin, xmax, xmin], 
               [xmin, xmin, xmax]]
        bc = [DirichletZero(), DirichletZero(), DirichletZero()]
        system = FlexibleSystem(atoms, box, bc)

    end
    
    return [system, energy, forces]
end

