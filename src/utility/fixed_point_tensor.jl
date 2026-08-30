"""
    fixed_point_tensor(T; nstates = 3, eig_tol = 1.0e-12, eig_krylovdim = 40,
                       return_basis = false)
    fixed_point_tensor(scheme::SLoopTNR; kwargs...)

Compute the normalized fixed-point tensor elements of a four-leg tensor `T` in
the transfer-matrix eigenbasis. The construction follows Eq. (2) of
[Ueda and Yamazaki (2023)](https://arxiv.org/abs/2307.02523): the four-tensor
`2 × 2` transfer matrices in the horizontal and vertical directions are
diagonalized, their leading `nstates` eigenvectors are used as boundary
projectors for a mirrored `2 × 2` tensor patch, and the result is normalized by
its `(1, 1, 1, 1)` element. The returned leg order is left-bottom-top-right,
matching the TNRKit convention.

For the critical Ising model, the three leading states are ordered as
`(1, σ, ε)`, so `fixed_point_tensor(scheme)[2, 2, 2, 2]` is the `σσσσ`
element.

Set `return_basis = true` to return a named tuple containing `elements`, the
horizontal and vertical bases, and their corresponding transfer-matrix
eigenvalues.
"""
function fixed_point_tensor(
        T::AbstractTensorMap{E, S, 4, 0}; nstates::Int = 3,
        eig_tol::Real = 1.0e-13, eig_krylovdim::Int = 100,
        return_basis::Bool = false
    ) where {E, S}
    nstates > 0 || throw(ArgumentError("nstates must be positive"))
    eig_tol > 0 || throw(ArgumentError("eig_tol must be positive"))
    eig_krylovdim > nstates || throw(ArgumentError("eig_krylovdim must exceed nstates"))

    A = convert(Array, T)
    allequal(size(A)) || throw(DimensionMismatch("all four tensor legs must have equal dimension"))
    nstates <= size(A, 1)^2 || throw(
        DimensionMismatch(
            "cannot retain $nstates states from a transfer matrix of dimension $(size(A, 1)^2)"
        )
    )

    horizontal_basis, horizontal_eigenvalues = _fixed_point_basis(
        T, true, nstates, eig_tol, eig_krylovdim
    )
    vertical_basis, vertical_eigenvalues = _fixed_point_basis(
        T, false, nstates, eig_tol, eig_krylovdim
    )
    horizontal_projector = conj.(horizontal_basis)
    vertical_projector = conj.(vertical_basis)
    Aflip = convert(Array, flip(T, (1, 2, 3, 4)))

    # TNRKit orders the legs as left-bottom-top-right. The four tensors are
    # related by mirror symmetry and arranged as
    #
    #                   top
    #              u -------- v
    #             /            \
    #       left a--T--------Tf--b right
    #              |          |
    #            c--Tf-------T--d
    #             \            /
    #              w -------- r
    #                  bottom
    #
    # and each pair of boundary indices is projected onto the corresponding
    # transfer-matrix eigenbasis. Keeping this as one optimized contraction
    # avoids materializing the eight-index boundary tensor.
    @tensoropt elements[left, bottom, top, right] :=
        A[a, x, u, y] * Aflip[b, z, v, y] * Aflip[c, x, w, q] * A[d, z, r, q] *
        horizontal_projector[a, c, left] * vertical_projector[w, r, bottom] *
        vertical_projector[u, v, top] * horizontal_projector[b, d, right]

    normalization = elements[1, 1, 1, 1]
    iszero(normalization) && throw(ArgumentError("the fixed-point identity element is zero"))
    elements ./= normalization

    if return_basis
        return (;
            elements, horizontal_basis, vertical_basis,
            horizontal_eigenvalues, vertical_eigenvalues,
        )
    end
    return elements
end

fixed_point_tensor(scheme::SLoopTNR; kwargs...) = fixed_point_tensor(scheme.T; kwargs...)

"""
    fixed_point_tensors(scheme::SLoopTNR; kwargs...)
    fixed_point_tensors(S, T; kwargs...)

Compute both the normalized four-leg `T` and three-leg `S` fixed-point tensor
elements. `S` is the optimized three-leg tensor whose four-fold contraction
produces `T`.

Following the upper construction in Fig. 6 of Ueda and Yamazaki (2023), four
optimized three-leg tensors form an open chain. The first two paired boundary
legs reuse the transfer projector of the two-`S` channel `StoSS(S)`, while the
pair at the two ends of the chain reuses the vertical projector of the
renormalized `T`. Thus no independent gauge is chosen for `S`: all three
projectors are the ones belonging to the corresponding four-leg `T` tensors
on the two sides of the renormalization step.

The returned named tuple contains `T`, `S`, the horizontal and vertical bases
of `T`, and `S_pair_basis`, the basis reused on the first two `S` legs. Both
tensors are normalized by their all-identity element. The `T` leg order is
left-bottom-top-right; the `S` indices follow Eq. (3) of Ueda and Yamazaki
(2023), with the equal pair of states first and the coarse state third.

At least one [`step!`](@ref) must have been applied to `scheme`, so that its
optimized three-leg tensor is available. Its internal and external bond
dimensions must also have reached the same retained dimension, as they do at
the fixed point.
"""
function fixed_point_tensors(
        scheme::SLoopTNR; nstates::Int = 3,
        eig_tol::Real = 1.0e-13, eig_krylovdim::Int = 100
    )
    isnothing(scheme.s_tensor) && throw(
        ArgumentError(
            "fixed_point_tensors requires an SLoopTNR scheme after at least one step"
        )
    )
    return fixed_point_tensors(
        scheme.s_tensor, scheme.T; nstates, eig_tol, eig_krylovdim
    )
end

function fixed_point_tensors(
        S_tensor::AbstractTensorMap{E, S, 2, 1},
        T::AbstractTensorMap{E, S, 4, 0}; nstates::Int = 3,
        eig_tol::Real = 1.0e-13, eig_krylovdim::Int = 100
    ) where {E, S}
    result = fixed_point_tensor(
        T; nstates, eig_tol, eig_krylovdim, return_basis = true
    )
    pair_tensor = StoSS(S_tensor)
    S_pair_basis, S_pair_eigenvalues = _fixed_point_basis(
        pair_tensor, true, nstates, eig_tol, eig_krylovdim
    )
    S_elements = _fixed_point_s_tensor(
        S_tensor, conj.(S_pair_basis), conj.(result.vertical_basis)
    )
    return (;
        T = result.elements, S = S_elements,
        result.horizontal_basis, result.vertical_basis,
        result.horizontal_eigenvalues, result.vertical_eigenvalues,
        S_pair_basis, S_pair_eigenvalues,
    )
end

function _fixed_point_s_tensor(S_tensor, horizontal_projector, vertical_projector)
    S = convert(Array, S_tensor)
    d1, d2, d3 = size(S)
    d1 == d2 || throw(
        DimensionMismatch(
            "the two internal S-tensor legs must have equal dimension"
        )
    )
    size(horizontal_projector, 1) == d1 &&
        size(horizontal_projector, 2) == d2 || throw(
        DimensionMismatch(
            "the two-S-channel projector does not match the internal S-tensor legs"
        )
    )
    size(vertical_projector, 1) == d3 &&
        size(vertical_projector, 2) == d3 || throw(
        DimensionMismatch(
            "the vertical T projector does not match the external S-tensor leg"
        )
    )

    # This is the upper tensor network in Fig. 6 with a projector on every
    # double boundary line:
    #
    #                j--P_p--m       l--P_q--n
    #                    |               |
    #             a--S---S-------S-------S--c
    #                 i       b       k
    #             \_________________________/
    #                         P_r
    #
    # TensorOperations chooses the same factorization through
    # M[l,m,j,k] = sum_i S[i,j,k] S[i,m,l] used in the paper's reference
    # implementation, without making M part of the public construction.
    @tensoropt elements[p, q, r] :=
        S[i, m, a] * S[i, j, b] * S[k, n, b] * S[k, l, c] *
        horizontal_projector[m, j, p] * horizontal_projector[n, l, q] *
        vertical_projector[a, c, r]

    normalization = elements[1, 1, 1]
    iszero(normalization) && throw(ArgumentError("the fixed-point S identity element is zero"))
    return elements ./ normalization
end

"""
    fixed_point_tensor_4x4(T; nstates = 3, eig_tol = 1.0e-12,
                          eig_krylovdim = 40, return_basis = false)
    fixed_point_tensor_4x4(scheme::SLoopTNR; kwargs...)

Compute fixed-point tensor elements from a mirrored `4 × 4` patch. Each CFT
basis state is an eigenvector on four boundary bonds. The horizontal and
vertical transfer matrices are applied matrix-free as four successive column
or row tensor-network contractions, so a dense matrix of size `D^4 × D^4` is
never constructed.

The result has left-bottom-top-right leg order and is normalized by its
`(1, 1, 1, 1)` element. With `return_basis = true`, the return value has the
same named-tuple layout as [`fixed_point_tensor`](@ref), but each basis has
shape `D × D × D × D × nstates`.
"""
function fixed_point_tensor_4x4(
        T::AbstractTensorMap{E, S, 4, 0}; nstates::Int = 3,
        eig_tol::Real = 1.0e-12, eig_krylovdim::Int = 40,
        return_basis::Bool = false
    ) where {E, S}
    nstates > 0 || throw(ArgumentError("nstates must be positive"))
    eig_tol > 0 || throw(ArgumentError("eig_tol must be positive"))
    eig_krylovdim > nstates || throw(ArgumentError("eig_krylovdim must exceed nstates"))

    A = convert(Array, T)
    allequal(size(A)) || throw(DimensionMismatch("all four tensor legs must have equal dimension"))
    d = size(A, 1)
    nstates <= d^4 || throw(
        DimensionMismatch(
            "cannot retain $nstates states from a transfer matrix of dimension $(d^4)"
        )
    )

    horizontal_basis, horizontal_eigenvalues = _fixed_point_basis_4x4(
        T, true, nstates, eig_tol, eig_krylovdim
    )
    vertical_basis, vertical_eigenvalues = _fixed_point_basis_4x4(
        T, false, nstates, eig_tol, eig_krylovdim
    )
    elements = _project_fixed_point_patch_4x4(
        A, convert(Array, flip(T, (1, 2, 3, 4))),
        conj.(horizontal_basis), conj.(vertical_basis)
    )

    normalization = elements[1, 1, 1, 1]
    iszero(normalization) && throw(ArgumentError("the fixed-point identity element is zero"))
    elements ./= normalization

    if return_basis
        return (;
            elements, horizontal_basis, vertical_basis,
            horizontal_eigenvalues, vertical_eigenvalues,
        )
    end
    return elements
end

fixed_point_tensor_4x4(scheme::SLoopTNR; kwargs...) =
    fixed_point_tensor_4x4(scheme.T; kwargs...)

function _fixed_point_transfer_matrix(
        T::AbstractTensorMap{E, S, 4, 0}, horizontal::Bool
    ) where {E, S}
    Tflip = flip(T, (1, 2, 3, 4))
    if horizontal
        @tensoropt transfer[-1 -2; -3 -4] :=
            T[-1 1; 3 2] * Tflip[-3 4; 5 2] *
            Tflip[-2 1; 3 6] * T[-4 4; 5 6]
    else
        @tensoropt transfer[-1 -2; -3 -4] :=
            T[1 3; -1 2] * Tflip[1 4; -2 2] *
            Tflip[5 3; -3 6] * T[5 4; -4 6]
    end
    dense_transfer = convert(Array, transfer)
    return reshape(
        dense_transfer, size(dense_transfer, 1) * size(dense_transfer, 2), :
    )
end

function _fixed_point_basis(
        T::AbstractTensorMap{E, S, 4, 0}, horizontal::Bool, nstates::Int,
        eig_tol::Real, eig_krylovdim::Int
    ) where {E, S}
    transfer = _fixed_point_transfer_matrix(T, horizontal)
    hermitian_transfer = Hermitian((transfer + transfer') / 2)
    x0 = convert.(eltype(transfer), sin.(eachindex(axes(transfer, 1))))
    eigenvalues, eigenvectors, info = eigsolve(
        hermitian_transfer, x0, nstates, :LM;
        krylovdim = min(size(transfer, 1), eig_krylovdim), maxiter = 300,
        tol = eig_tol, verbosity = 0
    )
    info.converged < nstates && @warn "Fixed-point transfer-matrix eigensolver did not converge" horizontal info

    order = sortperm(real.(eigenvalues); rev = true)[1:nstates]
    eigenvalues = eigenvalues[order]
    eigenvectors = reduce(hcat, eigenvectors[order])

    # Fix the otherwise arbitrary phase of every state. This makes tensor
    # elements with an odd number of a given state reproducible as well.
    for state in axes(eigenvectors, 2)
        vector = @view eigenvectors[:, state]
        pivot = vector[argmax(abs.(vector))]
        iszero(pivot) || (vector .*= conj(pivot) / abs(pivot))
    end

    d = dim(codomain(T)[1])
    return reshape(eigenvectors, d, d, nstates), eigenvalues
end

function _fixed_point_transfer_action_4x4(
        T::AbstractTensorMap{E, S, 4, 0}, horizontal::Bool
    ) where {E, S}
    A = convert(Array, T)
    Aflip = convert(Array, flip(T, (1, 2, 3, 4)))
    d = size(A, 1)

    if horizontal
        return function (vector)
            boundary = reshape(vector, d, d, d, d)
            for column in 4:-1:1
                boundary = _apply_fixed_point_column_4x4(boundary, A, Aflip, column)
            end
            return vec(boundary)
        end
    end
    return function (vector)
        boundary = reshape(vector, d, d, d, d)
        for row in 4:-1:1
            boundary = _apply_fixed_point_row_4x4(boundary, A, Aflip, row)
        end
        return vec(boundary)
    end
end

function _apply_fixed_point_row_4x4(boundary, A, Aflip, row::Int)
    tensors = [iseven(row + column) ? A : Aflip for column in 1:4]
    if isodd(row)
        indices = [
            [4, 5, -1, 1], [2, 6, -2, 1],
            [2, 7, -3, 3], [4, 8, -4, 3], [5, 6, 7, 8],
        ]
    else
        indices = [
            [4, -1, 5, 1], [2, -2, 6, 1],
            [2, -3, 7, 3], [4, -4, 8, 3], [5, 6, 7, 8],
        ]
    end
    return ncon([tensors..., boundary], indices)
end

function _apply_fixed_point_column_4x4(boundary, A, Aflip, column::Int)
    tensors = [iseven(row + column) ? A : Aflip for row in 1:4]
    if iseven(column)
        indices = [
            [5, 1, 4, -1], [6, 1, 2, -2],
            [7, 3, 2, -3], [8, 3, 4, -4], [5, 6, 7, 8],
        ]
    else
        indices = [
            [-1, 1, 4, 5], [-2, 1, 2, 6],
            [-3, 3, 2, 7], [-4, 3, 4, 8], [5, 6, 7, 8],
        ]
    end
    return ncon([tensors..., boundary], indices)
end

function _fixed_point_basis_4x4(
        T::AbstractTensorMap{E, S, 4, 0}, horizontal::Bool, nstates::Int,
        eig_tol::Real, eig_krylovdim::Int
    ) where {E, S}
    d = dim(codomain(T)[1])
    transfer = _fixed_point_transfer_action_4x4(T, horizontal)
    x0 = convert.(E, sin.(1:(d^4)))
    eigenvalues, eigenvectors, info = eigsolve(
        transfer, x0, nstates, :LM;
        krylovdim = min(d^4, eig_krylovdim), maxiter = 300,
        tol = eig_tol, verbosity = 0
    )
    info.converged < nstates && @warn "4 × 4 fixed-point transfer-matrix eigensolver did not converge" horizontal info

    order = sortperm(real.(eigenvalues); rev = true)[1:nstates]
    eigenvalues = eigenvalues[order]
    eigenvectors = reduce(hcat, eigenvectors[order])
    for state in axes(eigenvectors, 2)
        vector = @view eigenvectors[:, state]
        pivot = vector[argmax(abs.(vector))]
        iszero(pivot) || (vector .*= conj(pivot) / abs(pivot))
    end
    return reshape(eigenvectors, d, d, d, d, nstates), eigenvalues
end

function _project_fixed_point_patch_4x4(A, Aflip, horizontal_projector, vertical_projector)
    site_indices = [zeros(Int, 4) for _ in 1:4, _ in 1:4]
    label = 1

    for row in 1:4, column in 1:3
        leg = isodd(column) ? 4 : 1
        site_indices[row, column][leg] = label
        site_indices[row, column + 1][leg] = label
        label += 1
    end
    for row in 1:3, column in 1:4
        leg = isodd(row) ? 2 : 3
        site_indices[row, column][leg] = label
        site_indices[row + 1, column][leg] = label
        label += 1
    end

    left = Int[]
    right = Int[]
    for row in 1:4
        push!(left, label)
        site_indices[row, 1][1] = label
        label += 1
        push!(right, label)
        site_indices[row, 4][1] = label
        label += 1
    end
    bottom = Int[]
    top = Int[]
    for column in 1:4
        push!(bottom, label)
        site_indices[4, column][3] = label
        label += 1
        push!(top, label)
        site_indices[1, column][3] = label
        label += 1
    end

    tensors = Any[
        iseven(row + column) ? A : Aflip for row in 1:4 for column in 1:4
    ]
    indices = [site_indices[row, column] for row in 1:4 for column in 1:4]
    append!(
        tensors,
        (horizontal_projector, vertical_projector, vertical_projector, horizontal_projector)
    )
    append!(indices, ([left; -1], [bottom; -2], [top; -3], [right; -4]))
    return ncon(tensors, indices)
end
