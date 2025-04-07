using Revise
using Dagger
using DaggerGPU, CUDA
using LinearAlgebra
using KernelFunctions
using Distances
using Plots
using ScopedValues

function tile_precision(A, global_norm, scalar_factor, tolerance)

    tile_sqr = mapreduce(LinearAlgebra.norm_sqr, +, A)

    tile_norm = sqrt(tile_sqr)

    cal = tile_norm * scalar_factor / global_norm
    decision_hp = tile_norm * scalar_factor / global_norm < tolerance / eps(Float16)
    decision_sp = tile_norm * scalar_factor / global_norm < tolerance / eps(Float32)

    #We are planning in near future to support fp8  E4M3 and E5M2 
    #decision_fp8 = tile_norm * scalar_factor / global_norm < tolerance / 0.0625
    #if decision_fp8
    #    return Float8
    if decision_hp
        return Float16
    elseif decision_sp
        return Float32
    else
        return Float64
    end
end


function adapt_precision(A::DArray{T,2}, tolerance::T) where {T}

    Ac = parent(A).chunks
    mt, nt = size(Ac)

    global_norm = LinearAlgebra.norm2(A)

    MP = fill(T, mt, nt)
    DMP = view(MP, Blocks(1, 1))
    MPc = DMP.chunks


    for m in range(1, mt)
        for n in range(1, nt)
            if m!=n
                MPc[m, n] =
                    Dagger.@spawn tile_precision(
                        Ac[m, n],
                        global_norm, 
                        max(mt, nt), 
                        tolerance)
            end
        end
    end

    return collect(DMP)
end

@inline function mixedtrsm!(side, uplo, trans, diag, alpha, A, B, StoragePrecision)
    T = StoragePrecision
    if T == Float16
        T = Float32
    end
    m, n = size(B)
    if typeof(B) != CuMatrix{T}
        if typeof(A) != CuMatrix{T}
            Acopy = convert(CuMatrix{T}, A)
        else
            Acopy = A
        end
        Bcopy = convert(CuMatrix{T}, B)
        CUBLAS.trsm!(side, uplo, trans, diag, T(alpha), Acopy, Bcopy)
        copyto!(B, Bcopy)
        return B
    end
    BLAS.trsm!(side, uplo, trans, diag, alpha, A, B)
    return B
end
@inline function mixedgemm!(transa, transb, alpha, A, B, beta, C, StoragePrecision)
    T = StoragePrecision
    m, n = size(C)
    #if typeof(A) == CuMatrix{T} || typeof(B) == CuMatrix{T} || typeof(C) == CuMatrix{T} || typeof(A) == CuArray{T,2} || typeof(B) == CuArray{T,2} || typeof(C) == CuArray{T,2}
    #    println("GPU gemm")
    #else
    #    println("CPU gemm")
    #end
    #@show typeof(A), typeof(B), typeof(C)
   if typeof(C) != CuMatrix{T}
        if typeof(A) != CuMatrix{T}
            Acopy = convert(CuMatrix{T}, A)
        else
            Acopy = A
        end
        if typeof(B) != CuMatrix{T}
            Bcopy = convert(CuMatrix{T}, B)
        else
            Bcopy = B
        end
        if T == Float16
            Ccopy = convert(CuMatrix{Float32}, C)
        else
            Ccopy = convert(CuMatrix{T}, C)
        end
        #CUBLAS.gemm!(transa, transb, T(alpha), Acopy, Bcopy, T(beta), Ccopy)
        #LinearAlgebra.generic_matmatmul!(Ccopy, transa, transb, Acopy, Bcopy, LinearAlgebra.MulAddMul(T(alpha), T(beta)))
        CUBLAS.gemmEx!(transa, transb, alpha,Acopy,Bcopy, beta,Ccopy)
        copyto!(C, Ccopy)
        return C
    end
    #BLAS.gemm!(transa, transb, alpha, A, B, beta, C)
    LinearAlgebra.generic_matmatmul!(C, transa, transb, A, B, LinearAlgebra.MulAddMul(alpha, beta))
    return C
    
end
@inline function mixedsyrk!(uplo, trans, alpha, A, beta, C, StoragePrecision)
    T = StoragePrecision
    m, n = size(C)
    if typeof(C) != CuMatrix{T}
        if typeof(A) != CuMatrix{T}
            Acopy = convert(CuMatrix{T}, A)
        else
            Acopy = A
        end
        Ccopy = convert(CuMatrix{T}, C)
        CUBLAS.syrk!(uplo, trans, T(alpha), Acopy, T(beta), Ccopy)
        copyto!(C, Ccopy)
        return C
    end
    BLAS.syrk!(uplo, trans, alpha, A, beta, C)
    return C
end
@inline function mixedherk!(uplo, trans, alpha, A, beta, C, StoragePrecision)
    T = StoragePrecision
   if typeof(C) != CuMatrix{T}
        if typeof(A) != CuMatrix{T}
            Acopy = convert(CuMatrix{T}, A)
        else
            Acopy = A
        end
        Ccopy = convert(CuMatrix{T}, C)
        CUBLAS.herk!(uplo, trans, T(alpha), Acopy, T(beta), Ccopy)
        copyto!(C, Ccopy)
        return C
    end
    BLAS.herk!(uplo, trans, alpha, A, beta, C)
    return C
end
function MixedPrecisionChol!(A::DMatrix{T}, ::Type{LowerTriangular}, MP::Matrix{DataType}) where T
    LinearAlgebra.checksquare(A)

    zone = one(T)
    mzone = -one(T)
    rzone = one(real(T))
    rmzone = -one(real(T))
    uplo = 'L'
    Ac = A.chunks
    mt, nt = size(Ac)
    iscomplex = T <: Complex
    trans = iscomplex ? 'C' : 'T'


    info = [convert(LinearAlgebra.BlasInt, 0)]
    try
        Dagger.spawn_datadeps() do
            for k in range(1, mt)
                #Dagger.@spawn potrf_checked!(uplo, InOut(Ac[k, k]), Out(info))
                Dagger.@spawn Dagger.potrf_checked!(uplo, InOut(Ac[k, k]), Out(info))
                for m in range(k+1, mt)
                    Dagger.@spawn mixedtrsm!('R', uplo, trans, 'N', zone, In(Ac[k, k]), InOut(Ac[m, k]), MP[m,k])
                end
                for n in range(k+1, nt)
                    if iscomplex
                        Dagger.@spawn mixedherk!(uplo, 'N', rmzone, In(Ac[n, k]), rzone, InOut(Ac[n, n]), MP[n,n])
                    else
                        Dagger.@spawn mixedsyrk!(uplo, 'N', rmzone, In(Ac[n, k]), rzone, InOut(Ac[n, n]), MP[n,n])
                    end
                    for m in range(n+1, mt)
                        Dagger.@spawn mixedgemm!('N', trans, mzone, In(Ac[m, k]), In(Ac[n, k]), zone, InOut(Ac[m, n]), MP[m,n])
                    end
                end
            end
        end
    catch err
        err isa ThunkFailedException || rethrow()
        err = Dagger.Sch.unwrap_nested_exception(err.ex)
        err isa PosDefException || rethrow()
    end

    return LowerTriangular(A), info[1]
end

function origin_chol(A::DArray{T,2}, ::Type{LowerTriangular}) where T
    LinearAlgebra.checksquare(A)

    zone = one(T)
    mzone = -one(T)
    rzone = one(real(T))
    rmzone = -one(real(T))
    uplo = 'L'
    Ac = A.chunks
    mt, nt = size(Ac)
    iscomplex = T <: Complex
    trans = iscomplex ? 'C' : 'T'

    info = [convert(LinearAlgebra.BlasInt, 0)]
    try
        Dagger.spawn_datadeps() do
            for k in range(1, mt)
                Dagger.@spawn Dagger.potrf_checked!(uplo, InOut(Ac[k, k]), Out(info))
                for m in range(k+1, mt)
                    Dagger.@spawn BLAS.trsm!('R', uplo, trans, 'N', zone, In(Ac[k, k]), InOut(Ac[m, k]))
                end
                for n in range(k+1, nt)
                    if iscomplex
                        Dagger.@spawn BLAS.herk!(uplo, 'N', rmzone, In(Ac[n, k]), rzone, InOut(Ac[n, n]))
                    else
                        Dagger.@spawn BLAS.syrk!(uplo, 'N', rmzone, In(Ac[n, k]), rzone, InOut(Ac[n, n]))
                    end
                    for m in range(n+1, mt)
                        Dagger.@spawn BLAS.gemm!('N', trans, mzone, In(Ac[m, k]), In(Ac[n, k]), zone, InOut(Ac[m, n]))
                    end
                end
            end
        end
    catch err
        err isa DTaskFailedException || rethrow()
        err = Dagger.Sch.unwrap_nested_exception(err.ex)
        err isa PosDefException || rethrow()
    end

    return LowerTriangular(A), info[1]
end

function convert_to_mixed_precision(A::DArray{T,2},  MP::Matrix{DataType}) where {T}

end
function demo(N, nb)
    BLAS.set_num_threads(1)
    Dagger.MemPool.MEM_RESERVED[] = 0

    k = GammaExponentialKernel(; γ=0.5, metric=Euclidean())
    X = rand(N, N)
    A = kernelmatrix(k, X)
    
    A[diagind(A)] .+= 1
    CopyA = copy(A)

    bench = String[]
    ts = Float64[]

    DA_plain = distribute(A, Blocks(nb, nb))
    GC.enable(false)
    println("Full precision")
    #t_full = @elapsed cholesky!(DA_plain).L
    t_full = @elapsed LinearAlgebra._chol!(DA_plain, LowerTriangular)
    A_plain = CopyA#LowerTriangular(collect(DA_plain))
    push!(bench, "Full precision")
    push!(ts, t_full)

    for atol in [1e-4]#[1e-6, 1e-4, 1e-2]#[1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 7e-8, 8e-8, 9e-8, 1e-7]
        DA_mixed = distribute(A, Blocks(nb, nb))
        T = eltype(A)
        println("Adapt precision with atol = $atol")
        MP = adapt_precision(DA_mixed, T(atol))
        display(MP)

        # Let GC run
        GC.enable(true)
        GC.enable(false)

        println("Mixed precision with atol = $atol")
        region_size = 32
        t_mixed = 0.0
        #Dagger.with_options(;scope=Dagger.scope(cuda_gpu=1, threads=:)) do
        #Dagger.with_options(;scope=Dagger.scope(:any)) do
        Dagger.with_options(;scope=Dagger.scope(cuda_gpu=1)) do    
            ScopedValues.with(Dagger.DATADEPS_REGION_SPLIT=>region_size,
            Dagger.DATADEPS_SCHEDULER=>:roundrobin) do
                #DA_mixed = distribute(A, Blocks(nb, nb))
                t_mixed = @elapsed MixedPrecisionChol!(DA_mixed, LowerTriangular, MP) #origin_chol(DA_mixed, LowerTriangular)   #MixedPrecisionChol!(DA_mixed, LowerTriangular, MP)
            end
        end
        
        A_mixed = LowerTriangular(collect(DA_mixed))
        push!(bench, "Mixed precision with atol = $atol")
        push!(ts, t_mixed)
        @show t_full, t_mixed
        A = collect(A_mixed)

        B = rand(N, 1)
        Bcopy = copy(B)

        BLAS.trsm!('L', 'L', 'N', 'N', (1.0), A, B) 
        BLAS.trsm!('L', 'L', 'T', 'N', (1.0), A, B)

        @show norm(Bcopy - CopyA * B)/norm(Bcopy) #/ norm(CopyA) #(( norm(CopyA) * norm(B) + norm(Bcopy)) * 5000 * eps(Float64))
        @show norm(Bcopy - CopyA * B)/ (norm(Bcopy)-norm(CopyA)*  norm(B))


        @show norm(A_mixed - A_plain) / norm(A_plain)
    end
    GC.enable(true)

    #p = plot(bench, ts, yscale=:log10, xrotation=45, title="Cholesky factorization", ylabel="Time (s)", xlabel="Precision", legend=:outertopright)
    #savefig(p, "cholesky_bench.png")

    return

    #return A, A_copy

    #=
    B = rand(1000, 1)
    B_copy = copy(B)

    BLAS.trsm!('L', 'L', 'N', 'N', 1.0, A, B)
    BLAS.trsm!('L', 'L', 'N', 'N', 1.0, A, B)

    @show norm(B_copy - A_copy * B)
    #@show norm(Bcopy - CopyA * B) / 1000 / (norm(Bcopy)-norm(CopyA)*  norm(B))
    =#
end

