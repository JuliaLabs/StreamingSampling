#!/bin/bash
# Loading the required module
# source /etc/profile

# Job Flags
#SBATCH -p mit_preemptable
#SBATCH --mem=1024G

#module load mpi/openmpi-5.0.5 # Supercloud
#module load julia/1.11.3 # Supercloud
module load openmpi/5.0.6 # Engaging
module load julia/1.10.4 # Engaging
# Install Julia packages
julia --project=./ -e 'import Pkg; Pkg.instantiate()'
# Run the script
julia --project=./ compare-samplers-hfo2-balanced-partial-stats.jl
