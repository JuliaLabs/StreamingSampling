abstract type Kernel end

# radial basis function kernel (squared exponential kernel)
mutable struct RBF <: Kernel
    α::Float64
    ℓ::Float64
    β::Float64
end

function RBF(;α = 1e-8, ℓ = 1.0, β = 1.0)
    RBF(α, ℓ, β)
end

struct DotProduct <: Kernel
    α::Int
end
function DotProduct(; α = 2)
    DotProduct(α)
end

function Compute_DotProduct_Kernel(
    A::AbstractMatrix{T},
    d::DotProduct,
    ) where {T<:Real}
    
    m = size(A, 1)
    matAA = A * A'
    C = T.(diag(matAA) .* ones(m, m))

    return (matAA ./ sqrt.(C .* C')).^d.α
end

function Compute_DotProduct_Kernel(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    d::DotProduct,
    ) where {T<:Real}
    
    m = size(A, 1)
    matAB = A * B
    C = T.(diag(matAB) .* ones(m, m))

    return (matAB ./ sqrt.(C .* C')).^2
end
#=
function Compute_Distance_Matrix(
    A::AbstractMatrix{T}) where {T<:Real}

    m = size(A, 1)
    matAA = A * A'
    C = T.(diag(matAA) .* ones(m, m))
    return ((-2) .* matAA) +  (C) + (C')

end
=#

function Compute_Distance_Matrix(
    A::AbstractMatrix{T}) where {T<:Real}

   G = A * A'                   # GEMM on GPU
   s = sum(abs2, A; dims=2)     # m×1 squared norms (on GPU)
   D = @. s + s' - 2G           # broadcast on GPU
   # Clamp tiny negatives from roundoff (also fixes diagonal if it's -0)
   D .= max.(D, zero(T))
   # Zero the diagonal without indexing (broadcast with UniformScaling I)
   D[diagind(D)] .*=0.0

   return D
end

function Gaussian_Kernel(
    A::AbstractMatrix{T},
     r::RBF) where {T<:Real}

     return  T(r.β) .*  exp.(-A ./ (2 * T(r.ℓ)^2))
end
