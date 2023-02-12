using JuMP
import HiGHS


function build_and_pass_bm(N::Int64, opt)
    @info "Build model"
    @time begin
        model = direct_model(opt)
        set_silent(model)
        set_string_names_on_creation(model, false)
        @variable(model, x[1:N])
        @constraint(model, c, 1.5 * x + x .>= 1)
        @objective(model, Min, sum(x))
    end
    @info "First optimize"
    @time optimize!(model)
    @info "Update"
    @time set_normalized_coefficient.(c, x, 3.0)
    @info "Re-optimize"
    @time optimize!(model)
    @info "Total"
    return model
end

GC.gc()
@time build_and_pass_bm(50_000, HiGHS.Optimizer());

function build_and_pass(N::Int64, opt)
    @info "Build model"
    @time begin
        model = direct_model(opt)
        set_silent(model)
        set_string_names_on_creation(model, false)
        @variable(model, x[1:N])
        @variable(model, p[1:N])
        @constraint(model, p .== x)
        @constraint(model, c, 1.5 .* p .+ x .>= 1)
        @objective(model, Min, sum(x))
    end
    @info "First optimize"
    @time optimize!(model)
    @info "Update"
    @time set_normalized_coefficient.(c, p, 2.0)
    @info "Re-optimize"
    @time optimize!(model)
    @info "Total"
    return model
end

GC.gc()
@time build_and_pass(50_000, HiGHS.Optimizer());






struct ParamVarLink
    parameter::VariableRef
    variable::VariableRef
end

struct ParametricConstraint2{S} <: AbstractConstraint
    f::AffExpr
    s::S
    links::Vector{ParamVarLink}
end
ParametricConstraint{S} = ParametricConstraint2


function JuMP.build_constraint(
    _error::Function,
    f::JuMP.GenericQuadExpr,
    set::MOI.AbstractScalarSet,
    ::Type{ParametricConstraint};
)
    affine = f.aff
    terms = f.terms

    model = nothing
    
    parametric_variables = ParamVarLink[]
    sizehint!(parametric_variables, length(terms))
    
    for (vars, coeff) in terms
        if isnothing(model)
            model = owner_model(vars.a)
        end

        v_a = get(model.ext[:parameter], vars.a, nothing)
        v_b = get(model.ext[:parameter], vars.b, nothing)

        if v_a !== nothing
            _x = @variable(model)
            @constraint(model, _x == coeff * vars.b)
            affine.terms[_x] = v_a
            push!(parametric_variables, ParamVarLink(vars.a, _x))
        elseif v_b !== nothing
            _x = @variable(model)
            @constraint(model, _x == coeff * vars.a)
            affine.terms[_x] = v_b
            push!(parametric_variables, ParamVarLink(vars.b, _x))
        else
            _error("no")
        end
    end

    new_set = typeof(set)(MOI.constant(set) - affine.constant)
    affine.constant = 0

    return ParametricConstraint(affine, new_set, parametric_variables)
end

function JuMP.add_constraint(
    model::Model,
    con::ParametricConstraint,
    name::String,
)
    constr = add_constraint(
        model,
        ScalarConstraint(con.f, con.s)
    )
    model.ext[:links][constr] = con.links

    return constr
end

function finalize_parameters(model)
    be = backend(model)
    for (constr, pvls) in _m.ext[:links]
        for pvl in pvls
            MOI.modify(be, constr.index, MOI.ScalarCoefficientChange(pvl.variable.index, _m.ext[:parameter][pvl.parameter]))
        end
    end
    model.is_model_dirty = true

    return optimize!(model; ignore_optimize_hook = true)
end

function parameter(model, value::Float64; fix::Bool=true)
    _v = @variable(model)
    model.ext[:parameter][_v] = value
    fix && fix(_v, value; force=true)
    return _v
end

macro pconstraint(args...)
    return esc(:($JuMP.@constraint($(args...), ParametricConstraint)))
end

macro parameter(args...)
    args = JuMP._reorder_parameters(args)
    flat_args, kw_args, _ = JuMP.Containers._extract_kw_args(args)
    kw_args = Dict(kw.args[1] => kw.args[2] for kw in kw_args)

    if get(kw_args, :fix, true)
        return esc(quote
            _var = $JuMP.@variable($(flat_args[1:(end-1)]...))
            $(flat_args[1]).ext[:parameter][_var] = $(flat_args[end])
            $JuMP.fix(_var, $(flat_args[end]); force=true)
            _var
        end)
    else
        return esc(quote
            _var = $JuMP.@variable($(flat_args[1:(end-1)]...))
            $(flat_args[1]).ext[:parameter][_var] = $(flat_args[end])
            _var
        end)
    end
end

function _set_value(parameter::VariableRef, value::Float64; fix::Bool=true)
    model = owner_model(parameter)
    if !haskey(model.ext[:parameter], parameter)
        error("nope!")
    end

    model.ext[:parameter][parameter] = value
    fix && JuMP.fix(parameter, value; force=true)
    return nothing
end
JuMP.set_value(parameter::VariableRef, value::Float64; fix::Bool=true) = _set_value(parameter, value; fix=fix)

_m = Model(HiGHS.Optimizer)
set_optimize_hook(_m, finalize_parameters)

_m.ext[:parameter] = Dict{VariableRef, Float64}()
_m.ext[:links] = Dict{ConstraintRef, Vector{ParamVarLink}}()

_x = @variable(_m)
_p = @parameter(_m, 1.0; fix=false)
_c = @pconstraint(_m, _p * _x >= 1)

@objective(_m, Min, _x + 0)
optimize!(_m)

_m.ext[:links]
print(_m)

_m.ext[:parameter][_p] = 0.0
optimize!(_m)




function build_and_pass(N::Int64, opt)
    @info "Build model"
    @time begin
        model = Model(opt)
        set_optimize_hook(model, finalize_parameters)
        model.ext[:parameter] = Dict{VariableRef, Float64}()
        model.ext[:links] = Dict{ConstraintRef, Vector{ParamVarLink}}()
        set_silent(model)
        set_string_names_on_creation(model, false)
        x = [@variable(model) for i in 1:N]

        p = [@parameter(model, 1.0; fix=false) for i in 1:N]
        c = [@pconstraint(model, p[i] * x[i] >= 1) for i in 1:N]
        # c = [@constraint(model, x[i] >= 1) for i in 1:N]

        @objective(model, Min, sum(x))
    end
    @info "First optimize"
    @time begin
        optimize!(model)
    end
    @info "Update"
    @time begin
        set_value.(p, 2.0; fix=false)
    end
    @info "Re-optimize"
    @time optimize!(model)
    @info "Total"
    return model
end

@time build_and_pass(1, HiGHS.Optimizer);

GC.gc()
@time build_and_pass(50_000, HiGHS.Optimizer);
