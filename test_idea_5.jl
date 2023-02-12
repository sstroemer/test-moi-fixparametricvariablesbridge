include("new_idea_5.jl")


_m = Model(HiGHS.Optimizer)
enable_parameters!(_m)

_x = @variable(_m)

_p = @variable(_m)
push!(_m.ext[:__parameter_values], 1.0)
_m.ext[:__parameters][_p] = ParamData(0x01)

_c = @constraint(_m, [i=1:50000], 2 * _p * _x >= 1, ParametricConstraint)

@objective(_m, Min, sum(_x))

set_optimize_hook(_m, _finalize_parameters)
optimize!(_m)


function build_and_pass(N::Int64, opt)
    @info "Build model"
    @time begin
        model = direct_model(opt)
        set_silent(model)
        set_string_names_on_creation(model, false)
        enable_parameters!(model)

        @variable(model, x[1:N])
        @variable(model, p[1:N])

        for i in 1:N
            model.ext[:__parameters][p[i]] = ParamData(UInt64(i), 1.0)
        end

        @constraint(model, c[i=1:N], 1.5 * p[i] * x[i] >= 1, ParametricConstraint)
        @objective(model, Min, sum(x))
    end
    @info "First optimize"
    @time optimize!(model)
    @info "Update"
    @time begin 
        for i in 1:N
            model.ext[:__parameters][p[i]].value = 2.0
        end
    end
    @info "Re-optimize"
    @time optimize!(model)
    @info "Total"
    return model
end

@time build_and_pass(1, Gurobi.Optimizer());

GC.gc()
@time _m = build_and_pass(50_000, Gurobi.Optimizer());

_m[:c][1]
_m.ext[:__parameters][_m[:p][1]].value = 4.0
_m.ext[:__links][1]

param_values = collect(it.value for it in values(_m.ext[:__parameters]))
eval_var_coeff(_m.ext[:__links][1].update)

@time optimize!(_m)


using ProfileView
ProfileView.@profview build_and_pass(50_000, Gurobi.Optimizer(