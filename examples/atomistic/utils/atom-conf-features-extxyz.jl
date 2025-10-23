# Reads an atomistic system: (EXTXYZ files)
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

        # Read header
        line = readline(io)
        
        # Read lattice
        lattice_line = match(r"Lattice=\"(.*?)\" ", line).captures[1]
        lattice = parse.(T, split(lattice_line)) * distance_units
        box = [lattice[1:3], lattice[4:6], lattice[7:9]]
        
        # Read energy
        energy = try
            energy_line = match(r"energy=(.*?) ", line).captures[1]
            energy = parse(T, energy_line)
        catch
            NaN
        end

        # Read boundary conditions
        bc = []
        try
            bc_line = match(r"pbc=\"(.*?)\"", line).captures[1]
            bc = [t == "T" ? Periodic() : DirichletZero() for t in split(bc_line)]
        catch
            bc = [DirichletZero(), DirichletZero(), DirichletZero()]
        end

        # Read properties
        properties = match(r"Properties=(.*?) ", line).captures[1]
        properties = split(properties, ":")
        properties = [properties[i:i+2] for i = 1:3:(length(properties)-1)]
        
        # Read atoms
        atoms = Vector{AtomsBase.Atom}(undef, num_atoms)
        for i = 1:num_atoms
            line = split(readline(io))
            line_count = 1
            position = 0.0
            element = 0.0
            data = Dict(())
            for prop in properties
                if prop[1] == "species"
                    element = Symbol(line[line_count])
                    line_count += 1
                elseif prop[1] == "pos"
                    position = SVector{3}(parse.(T, line[line_count:line_count+2]))
                    line_count += 3
                elseif prop[1] == "move_mask"
                    ft = Symbol(line[line_count])
                    line_count += 1
                elseif prop[1] == "tags"
                    ft = Symbol(line[line_count])
                    line_count += 1
                elseif prop[1] == "forces"
                    push!(forces, parse.(T, line[line_count:line_count+2]))
                    line_count += 3
                else
                    length = parse(Int, prop[3])
                    if length == 1
                        data = merge(data, Dict((Symbol(prop[1]) => line[line_count])))
                    else
                        data = merge(
                            data,
                            Dict((
                                Symbol(prop[1]) =>
                                    line[line_count:line_count+length-1]
                            )),
                        )
                    end
                end
            end
            if isempty(data)
                atoms[i] = AtomsBase.Atom(element, position .* distance_units)
            else
                atoms[i] =
                    AtomsBase.Atom(element, position .* distance_units, data...)
            end

        end

        system = FlexibleSystem(atoms, box, bc)

    end

    return [system, energy, forces]
end


