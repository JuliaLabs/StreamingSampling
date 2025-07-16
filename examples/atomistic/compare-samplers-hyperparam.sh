#!/bin/bash
# Loading the required module
source /etc/profile
module load mpi/openmpi-5.0.5
module load julia/1.11.3
# Install Julia packages
julia --project=./ -e 'import Pkg; Pkg.instantiate()'
# Run the script
julia --project=./ compare-samplers-hyperparam.jl
