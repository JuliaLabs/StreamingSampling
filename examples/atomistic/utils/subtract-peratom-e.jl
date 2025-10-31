function subtract_peratom_e(config::Configuration, vref_dict)
    orig_e = get_energy(config)
    e_unit = orig_e.u
    new_e = get_values(orig_e)
    for atom in get_system(config).particles
        species = atom.atomic_symbol
        new_e -= vref_dict[species]
    end

    Energy(new_e,e_unit)
end

function adjust_energies!(ds, vref_dict)
    for config in ds
        new_energy = subtract_peratom_e(config,vref_dict)
        config.data[Energy] = new_energy
    end
end

function get_all_energies_w_onebody(
    ds::DataSet,
    bp::BasisPotential,
    vref_dict
)
    energies = Float64[]
    for config in ds
        energy = PotentialLearning.potential_energy(config, bp)
        for atom in get_system(config).particles
            species = atom.atomic_symbol
            energy += vref_dict[species]
        end
        push!(energies, energy)
    end

    return energies
end

function get_all_energies_w_onebody(
    ds::DataSet,
    vref_dict
)
    energies = Float64[]
    for config in ds
        energy = get_values(get_energy(config))
        for atom in get_system(config).particles
            species = atom.atomic_symbol
            energy += vref_dict[species]
        end
        push!(energies, energy)
    end

    return energies
end

