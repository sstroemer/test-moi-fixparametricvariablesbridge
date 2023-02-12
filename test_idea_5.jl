import HiGHS
import Gurobi
using JuMP
# include("new_idea_5.jl")



function build_and_pass_bm(N::Int64, opt)
    @info "Build model"
    @time begin
        model = direct_model(opt)
        set_silent(model)
        set_string_names_on_creation(model, false)

        @variable(model, x[1:N])
        @constraint(model, c[i=1:N], 1.5 * x[i] >= 1)
        # @constraint(model, b[i=1:N], x[i] >= (mod(i, 2) == 0 ? 5 : 0))
        @objective(model, Min, sum(x))
    end
    @info "First optimize"
    @time optimize!(model)
    @info "Update"
    @time begin 
        set_normalized_coefficient.(c, x, (2.0 * 1.5) .* ones(N))
    end
    @info "Re-optimize"
    @time optimize!(model)
    @info "Total"
    return model
end

function build_and_pass(N::Int64, opt)
    @info "Build model"
    @time begin
        model = direct_model(opt)
        set_silent(model)
        set_string_names_on_creation(model, false)
        enable_parameters!(model)

        @variable(model, x[1:N])
        # @parameter(model, p[1:N], ones(N); fix=false)  # <-- performance is bad
        @variable(model, p[1:N])
        for i in 1:N
            model.ext[:__parameters][p[i]] = ParameterData(UInt64(i), 1.0)
        end

        @pconstraint(model, c[i=1:N], 1.5 * p[i] * x[i] >= 1)
        # @constraint(model, b[i=1:N], x[i] >= (mod(i, 2) == 0 ? 5 : 0))
        @objective(model, Min, sum(x))
    end
    @info "First optimize"
    @time optimize!(model)
    @info "Update"
    @time begin 
        set_value.(p, 2.0 .* ones(N); fix=false)
    end
    @info "Re-optimize"
    @time optimize!(model)
    @info "Total"
    return model
end

@time build_and_pass(1, Gurobi.Optimizer());
@time build_and_pass_bm(1, Gurobi.Optimizer());

GC.gc()
@time _m = build_and_pass(50_000, Gurobi.Optimizer());
GC.gc()
@time _m = build_and_pass_bm(50_000, Gurobi.Optimizer());



using ProfileView
ProfileView.@profview build_and_pass(50_000, Gurobi.Optimizer())