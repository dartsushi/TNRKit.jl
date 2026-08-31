"""
$(TYPEDEF)

c4 & inversion symmetric Loop Optimization for Tensor Network Renormalization

# Constructors
    $(FUNCTIONNAME)(T)
    $(FUNCTIONNAME)(TA, TB)

# Running the algorithm
    run!(::SLoopTNR, trscheme::TruncationStrategy,
              criterion::TNRKit.stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, oneloop=true,
              verbosity=1])

`oneloop=true` will use disentangled tensors as a starting guess for the optimization.
# Fields

$(TYPEDFIELDS)

# References
* [Yang et. al. Phys. Rev. Letters 118 (2017)](@cite yang2017) (Fig. S6)

"""
mutable struct SLoopTNR{E, S, TT <: AbstractTensorMap{E, S, 4, 0}} <: TNRScheme{E, S}
    "Central tensor"
    T::TT

    "Optimized three-leg tensor from the latest renormalization step"
    s_tensor::Any

    "Gradient optimization algorithm"
    gradalg::OptimKit.LBFGS
    function SLoopTNR(T::TT; gradalg = LBFGS(10; verbosity = 0, gradtol = 6.0e-7, maxiter = 40000)) where {E, S, TT <: AbstractTensorMap{E, S, 4, 0}}
        if any(
                charge != dual(charge) for leg in 1:4 for
                    charge in sectors(space(T, leg))
            )
            throw(ArgumentError(
                "SLoopTNR currently requires self-dual symmetry sectors because its " *
                    "reflection-symmetric cost identifies every virtual space with " *
                    "its dual. Complex Z_q irreps are therefore unsupported for q > 2; " *
                    "use classical_potts_inv(Trivial, q) instead."
            ))
        end
        return new{E, S, TT}(T, nothing, gradalg)
    end
end

########## Initial tensor ##########
"""
    classical_potts_inv(q::Int, β::Real)
    classical_potts_inv(q::Int)
    classical_potts_inv(::Type{Trivial}, q::Int, β::Real)
    classical_potts_inv(::Type{ZNIrrep{N}}, q::Int, β::Real) where N

Construct the all-outgoing, reflection- and `C₄`-symmetric tensor for the
ferromagnetic `q`-state Potts model. If `β` is omitted, the critical inverse
temperature [`potts_βc(q)`](@ref) is used.

The default uses dense nonsymmetric legs, which are compatible with the
self-dual virtual spaces required by [`SLoopTNR`](@ref). An explicit
`ZNIrrep{q}` method is also provided for constructing the tensor in the charge
basis. For `q > 2`, however, the charges `a` and `-a` are distinct complex
irreps, so that tensor cannot currently be evolved by `SLoopTNR`.

In the explicit charge basis, the only nonzero entries satisfy

```math
a + b + c + d = 0 \\pmod q,
```

The standard [`classical_potts`](@ref) tensor is reoriented so that all four
legs are in the codomain, as required by [`SLoopTNR`](@ref).

# Examples
```julia
classical_potts_inv(3)                    # SLoopTNR-compatible dense tensor
classical_potts_inv(Trivial, 4, 1.0)      # custom temperature
classical_potts_inv(ZNIrrep{3}, 3, 1.0)   # explicit charge-basis tensor
```
"""
function _check_potts_inv_parameters(q::Int, β::Real)
    q >= 2 || throw(ArgumentError("the number of Potts states must be at least two"))
    β >= zero(β) || throw(ArgumentError("SLoopTNR requires ferromagnetic Potts coupling β >= 0"))
    return nothing
end

function classical_potts_inv(::Type{Trivial}, q::Int, β::Real)
    _check_potts_inv_parameters(q, β)
    tensor = permute(classical_potts(Trivial, q, β), (1, 2, 3, 4))
    return flip(tensor, (3, 4))
end

function classical_potts_inv(::Type{ZNIrrep{N}}, q::Int, β::Real) where {N}
    _check_potts_inv_parameters(q, β)
    N == q || throw(ArgumentError("number of irreps must match the number of Potts states"))
    tensor = permute(classical_potts(ZNIrrep{N}, q, β), (1, 2, 3, 4))
    return flip(tensor, (3, 4))
end
classical_potts_inv(q::Int, β::Real) = classical_potts_inv(Trivial, q, β)
function classical_potts_inv(::Type{Trivial}, q::Int)
    q >= 2 || throw(ArgumentError("the number of Potts states must be at least two"))
    return classical_potts_inv(Trivial, q, potts_βc(q))
end
function classical_potts_inv(::Type{ZNIrrep{N}}, q::Int) where {N}
    q >= 2 || throw(ArgumentError("the number of Potts states must be at least two"))
    return classical_potts_inv(ZNIrrep{N}, q, potts_βc(q))
end
function classical_potts_inv(q::Int)
    q >= 2 || throw(ArgumentError("the number of Potts states must be at least two"))
    return classical_potts_inv(q, potts_βc(q))
end

function classical_ising_inv(β)
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    T = zeros(Float64, S ⊗ S ← S' ⊗ S')
    block(T, Irrep[ℤ₂](0)) .= [2x^2 2x * y; 2x * y 2y^2]
    block(T, Irrep[ℤ₂](1)) .= [2x * y 2x * y; 2x * y 2x * y]

    return permute(T, (1, 2, 3, 4))
end
classical_ising_inv() = classical_ising_inv(ising_βc)

########## utility functions ##########
function trnorm_2x2(T)
    @tensoropt TT[-1 -2; -3 -4] := T[1 -1 2 -3] * conj(T[1 -2 2 -4])
    return sqrt(TTtoNorm(TT))
end

########## Cost function ##########
function StoSS(S)
    V = domain(S)[1]
    b = isomorphism(V, V')
    @tensoropt SS[-1 -2 -3 -4] := S[-1 -2; 1] * S[-3 -4; 2] * b[1 2]
    return SS
end

function TTtoNorm(TT)
    V = domain(TT)
    b = isomorphism(V[1] ⊗ V[2], V[1]' ⊗ V[2]')
    TTb = TT * b
    @tensoropt T4[-1 -2; -3 -4] := TT[-1 -2; 1 2] * TTb[-3 -4; 1 2]
    V = domain(T4)
    b = isomorphism(V[1] ⊗ V[2], V[1]' ⊗ V[2]')
    T4b = T4 * b
    @tensoropt T8[-1 -2; -3 -4] := T4[-1 -2; 1 2] * T4b[-3 -4; 1 2]
    V = domain(T8)
    b = isomorphism(V[1] ⊗ V[2], V[1]' ⊗ V[2]')
    return tr(T8 * b)
end

function TtoNorm(T)
    @tensoropt TT[-1 -2; -3 -4] := T[1 -1 2 -3] * conj(T[1 -2 2 -4])
    return TTtoNorm(TT)
end

function cost_looptnr(S, T, n_TT)
    @assert eltype(S) == Float64 "Modification is needed for complex numbers!"
    SS = StoSS(S)

    @tensoropt TSS[-1 -2; -3 -4] := T[1 -1 2 -3] * conj(SS[1 -2 2 -4])
    @tensoropt S4[-1 -2; -3 -4] := SS[1 -1 2 -3] * conj(SS[1 -2 2 -4])
    return n_TT + TTtoNorm(S4) - 2 * TTtoNorm(TSS)
end

########## Gradient Optimization ##########
function loop_environment(TT)
    V = domain(TT)
    b = isomorphism(V[1] ⊗ V[2], V[1]' ⊗ V[2]')
    TTb = TT * b
    @tensoropt T4[-1 -2; -3 -4] := TT[-1 -2; 1 2] * TTb[-3 -4; 1 2]
    return T4 * b * TT
end

function StoSS_pullback(S, gSS)
    V = domain(S)[1]
    b = isomorphism(V, V')
    Sb = S * b
    @tensoropt gS[-1 -2; -3] := gSS[-1 -2 1 2] * conj(Sb[1 2; -3])
    return 2 * gS
end

function gradient_looptnr(S, T, SS, TSS, S4)
    # One of the eight equal contributions from the S-S overlap.
    env_SS = loop_environment(S4)
    @tensoropt gSS[-1 -2 -3 -4] := SS[-1 1 -3 2] * env_SS[-2 1; -4 2]
    gS_SS = StoSS_pullback(S, gSS)

    # One of the four equal contributions from the T-S overlap.
    env_TS = loop_environment(TSS)
    env_TS_transposed = permute(env_TS, ((2, 1), (4, 3)))
    @tensoropt gTS_dual[-1 -2 -3 -4] := conj(T[-1 1 -3 2]) * env_TS_transposed[-2 1; -4 2]
    gTS = flip(gTS_dual, (1, 2, 3, 4))
    gS_TS = StoSS_pullback(S, gTS)

    return 8 * gS_SS - 2 * 4 * gS_TS
end

function cost_looptnr_fg(S, T, n_TT)
    @assert eltype(S) == Float64 "Modification is needed for complex numbers!"
    SS = StoSS(S)

    @tensoropt TSS[-1 -2; -3 -4] := T[1 -1 2 -3] * conj(SS[1 -2 2 -4])
    @tensoropt S4[-1 -2; -3 -4] := SS[1 -1 2 -3] * conj(SS[1 -2 2 -4])
    cost = n_TT + TTtoNorm(S4) - 2 * TTtoNorm(TSS)
    grad = gradient_looptnr(S, T, SS, TSS, S4)
    return cost, grad
end

function optimize_S(scheme, S)
    n_TT = TtoNorm(scheme.T)
    opt_fg(x) = cost_looptnr_fg(x, scheme.T, n_TT)
    Sopt, fx, gx, numfg, normgradhistory = optimize(
        opt_fg, S,
        scheme.gradalg
    )
    return Sopt
end

########## Entanglement filtering ##########
function Ψ_center(T)
    Tflip = flip(T, (1, 2, 3, 4))
    psi = [
        permute(T, ((2,), (1, 3, 4))),
        permute(Tflip, ((4,), (3, 1, 2))),
        permute(T, ((2,), (1, 3, 4))),
        permute(Tflip, ((4,), (3, 1, 2))),
    ]
    return psi
end

function Ψ_corner(T)
    Tflip = flip(T, (1, 2, 3, 4))
    psi = [
        permute(T, ((3,), (4, 2, 1))),
        permute(Tflip, ((1,), (2, 4, 3))),
        permute(T, ((3,), (4, 2, 1))),
        permute(Tflip, ((1,), (2, 4, 3))),
    ]
    return psi
end

function entanglement_filtering(T; trunc = trunctol(atol = 1.0e-12))
    entanglement_function(steps, data) = abs(data[end])
    entanglement_criterion = maxiter(200) & convcrit(1.0e-12, entanglement_function)

    psi_center = Ψ_center(T)
    psi_corner = Ψ_corner(T)

    PR_list, PL_list = TNRKit.find_projectors(
        psi_center, [1, 1, 1, 1], [3, 3, 3, 3],
        entanglement_criterion, trunc
    )
    P_bottom = PL_list[1]
    P_right = PL_list[1]

    PR_list, PL_list = TNRKit.find_projectors(
        psi_corner,
        [1, 1, 1, 1], [3, 3, 3, 3],
        entanglement_criterion, trunc
    )
    P_top = PL_list[3]
    P_left = PL_list[3]

    @tensoropt T_new[-1 -2 -3 -4] := T[1 2 3 4] * P_left[-1; 1] * P_bottom[-2; 2] *
        P_top[-3; 3] * P_right[-4; 4]
    return T_new
end

########## Initialization of loop optimizations ##########
function decompose_T(T, trunc)
    u, s, _ = svd_trunc(T, (1, 2), (3, 4); trunc = trunc)
    return u * sqrt(s)
end

function ef_oneloop(T, trunc::TruncationStrategy)
    ΨA = Ψ_center(T)
    ΨB = [s for A in ΨA for s in SVD12(A, truncrank(trunc.howmany * 2))]

    ΨB_function(steps, data) = abs(data[end])
    criterion = maxiter(100) & convcrit(1.0e-12, ΨB_function)
    PRs, _ = find_projectors(
        ΨB, [1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2],
        criterion, trunc
    )
    i = 1
    @tensoropt S[-2 -1; -3] := ΨB[i][-1; -2 2] * PRs[mod(i, 8) + 1][2; -3]
    return S
end

########## Updating the tensor ##########
function combine_4S(S)
    Sflip = flip(S, (1, 2))
    @tensoropt Tnew[-1 -2 -3 -4] := S[1 2; -4] * Sflip[1 4; -3] * S[3 4; -1] * Sflip[3 2; -2]
    return Tnew
end

########## Main funcitons ##########
function step!(scheme::SLoopTNR, trunc::TruncationStrategy, oneloop)
    scheme.T = entanglement_filtering(scheme.T)
    if oneloop == true
        S = ef_oneloop(scheme.T, trunc)
    else
        S = decompose_T(scheme.T, trunc)
    end
    S = optimize_S(scheme, S)
    scheme.s_tensor = S
    scheme.T = combine_4S(S)
    return scheme
end

function run!(
        scheme::SLoopTNR, trscheme::TruncationStrategy,
        criterion::TNRKit.stopcrit; finalizer = default_Finalizer, finalize_beginning = true, oneloop = true,
        verbosity = 1
    )
    data = output_type(finalizer)[]

    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"

        if finalize_beginning
            push!(data, finalizer.f!(scheme))
        end
        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            step!(scheme, trscheme, oneloop)
            push!(data, finalizer.f!(scheme))
            steps += 1
            crit = criterion(steps, data)
        end
        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end

function Base.show(io::IO, scheme::SLoopTNR)
    println(io, "Symmetric LoopTNR - C4 and reflection symmetric scheme")
    println(io, "  * T: $(summary(scheme.T))")
    isnothing(scheme.s_tensor) || println(io, "  * S: $(summary(scheme.s_tensor))")
    println(io, "  * gradalg: $(summary(scheme.gradalg))")
    return nothing
end
