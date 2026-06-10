using Pkg
using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "SurrogatesBase" begin
    if GROUP == "QA"
        Pkg.activate(joinpath(@__DIR__, "qa"))
        Pkg.develop(path = joinpath(@__DIR__, ".."))
        Pkg.instantiate()
        @safetestset "Quality Assurance" include("qa.jl")
    end

    if GROUP == "All" || GROUP == "Core"
        @safetestset "Core" include("core_tests.jl")
        @safetestset "Allocation Tests" include("alloc_tests.jl")
    end
end
