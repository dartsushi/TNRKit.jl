module TRGKit
    using TensorKit, LinearAlgebra
    using Printf

    # stop criteria
    include("utility/stopping.jl")
    export maxiter, convcrit

    # schemes
    include("schemes/trgscheme.jl")
    include("schemes/trg.jl")
    include("schemes/btrg.jl")
    include("schemes/hotrg.jl")
    
    export TRGScheme, TRG, BTRG, HOTRG, trg_convcrit, btrg_convcrit, hotrg_convcrit
    export run!

    # models
    include("models/ising.jl")
    export classical_ising

    include("models/gross-neveu.jl")
    export gross_neveu_start

    # export the different models (do we really want to ship those ourselves?)    
end 
