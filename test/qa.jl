using SurrogatesBase, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(SurrogatesBase)
    Aqua.test_ambiguities(SurrogatesBase, recursive = false)
    Aqua.test_deps_compat(SurrogatesBase)
    Aqua.test_piracies(SurrogatesBase)
    Aqua.test_project_extras(SurrogatesBase)
    Aqua.test_stale_deps(SurrogatesBase)
    Aqua.test_unbound_args(SurrogatesBase)
    Aqua.test_undefined_exports(SurrogatesBase)
end
