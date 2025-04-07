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

function Compute_Distance_Matrix(
    A::AbstractMatrix{T}) where {T<:Real}

    m = size(A, 1)
    matAA = A * A'
    C = T.(diag(matAA) .* ones(m, m))
    return ((-2) .* matAA) +  (C) + (C')

end

function Gaussian_Kernel(
    A::AbstractMatrix{T},
     r::RBF) where {T<:Real}

     return  T(r.β) .*  exp.(-A ./ (2 * T(r.ℓ)^2))
end