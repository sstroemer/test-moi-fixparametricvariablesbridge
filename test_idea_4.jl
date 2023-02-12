include("new_idea_4.jl")


_m = Model(HiGHS.Optimizer)
enable_parameters!(_m)

_x = @variable(_m)

_p = @variable(_m)
push!(_m.ext[:__parameter_values], 1.0)
_m.ext[:__parameters][_p] = ParamData(0x01)
push!(_m.ext[:__nodes], Node(Float64; feature=1))

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

        append!(model.ext[:__nodes], [Node(Float64; feature=i) for i in 1:N])
        append!(model.ext[:__parameter_values], ones(N))
        for i in 1:N
            model.ext[:__parameters][p[i]] = ParamData(UInt64(i))
        end

        @constraint(model, c[i=1:N], 1.5 * p[i] * x[i] >= 1, ParametricConstraint)
        @objective(model, Min, sum(x))
    end
    @info "First optimize"
    @time optimize!(model)
    @info "Update"
    @time model.ext[:__parameter_values] = 2 .* ones(N) 
    @info "Re-optimize"
    @time optimize!(model)
    @info "Total"
    return model
end

@time build_and_pass(1, HiGHS.Optimizer());

GC.gc()
@time _m = build_and_pass(50_000, HiGHS.Optimizer());
