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

mutable struct ParamData
    value::Number
    is_dirty::Bool
end

struct ParametricConstraint2{S} <: AbstractConstraint
    f::AffExpr
    s::S
    links::Vector{ParamVarLink}
end
ParametricConstraint{S} = ParametricConstraint2

function JuMP.build_constraint(
    _error::Function,
    f::JuMP.GenericAffExpr,
    set::MOI.AbstractScalarSet,
    ::Type{ParametricConstraint};
)
    return JuMP.build_constraint(_error, f, set)
end

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

        v_a = get(model.ext[:__parameters], vars.a, nothing)
        v_b = get(model.ext[:__parameters], vars.b, nothing)

        if v_a !== nothing
            _x = @variable(model)
            @constraint(model, _x == coeff * vars.b)
            affine.terms[_x] = v_a.value
            push!(parametric_variables, ParamVarLink(vars.a, _x))
        elseif v_b !== nothing
            _x = @variable(model)
            @constraint(model, _x == coeff * vars.a)
            affine.terms[_x] = v_b.value
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
    model.ext[:__links][constr] = con.links

    return constr
end

function _finalize_parameters(model::JuMP.Model)
    for (constr, pvls) in model.ext[:__links]
        for i in eachindex(model.ext[:__links][constr])
            !model.ext[:__parameters][pvls[i].parameter].is_dirty && continue
            set_normalized_coefficient(constr, pvls[i].variable, model.ext[:__parameters][pvls[i].parameter].value)
            # MOI.modify(be, constr.index, MOI.ScalarCoefficientChange(pvls[i].variable.index, model.ext[:__parameters][pvls[i].parameter].value))
        end
    end
    model.is_model_dirty = true

    return optimize!(model; ignore_optimize_hook = true)
end

# function parameter(model, value::Float64; fix::Bool=true)
#     _v = @variable(model)
#     model.ext[:__parameters][_v] = value
#     fix && fix(_v, value; force=true)
#     return _v
# end

macro pconstraint(args...)
    return esc(:($JuMP.@constraint($(args...), ParametricConstraint)))
end

macro parameter(args...)
    _error(str...) = JuMP._macro_error(:parameter, args, __source__, str...)

    args = JuMP._reorder_parameters(args)
    flat_args, kw_args, _ = JuMP.Containers._extract_kw_args(args)
    kw_args = Dict(kw.args[1] => kw.args[2] for kw in kw_args)

    if !(length(flat_args) in [2, 3])
        _error("Wrong number of arguments. Did you miss the initial parameter value?")
    end
    if !(flat_args[end] isa Number)
        _error("Wrong arguments. Did you miss the initial parameter value?")
    end
    _vector_scalar_get(_scalar::Number, ::Int64) = _scalar
    _vector_scalar_get(_vector::Vector{T} where T<:Number, i::Int64) = _vector[i]

    if get(kw_args, :fix, true)
        return esc(quote
            _var = $JuMP.@variable($(flat_args[1:(end-1)]...))
            if _var isa Vector
                for i in eachindex(_var)
                    $(flat_args[1]).ext[:__parameters][_var[i]] = ParamData($_vector_scalar_get($(flat_args[end]), i), false)
                end
            else
                $(flat_args[1]).ext[:__parameters][_var] = ParamData($(flat_args[end]), false)
                $JuMP.fix(_var, $(flat_args[end]); force=true)
            end
            _var
        end)
    else
        return esc(quote
            _var = $JuMP.@variable($(flat_args[1:(end-1)]...))
            if _var isa Vector
                for i in eachindex(_var)
                    $(flat_args[1]).ext[:__parameters][_var[i]] = ParamData($_vector_scalar_get($(flat_args[end]), i), false)
                end
            else
                $(flat_args[1]).ext[:__parameters][_var] = ParamData($(flat_args[end]), false)
            end
            _var
        end)
    end
end

function _set_value(parameter::VariableRef, value::Float64; fix::Bool=true)
    model = owner_model(parameter)
    if !haskey(model.ext[:__parameters], parameter)
        error("nope!")
    end

    model.ext[:__parameters][parameter].value = value
    model.ext[:__parameters][parameter].is_dirty = true
    fix && JuMP.fix(parameter, value; force=true)
    return nothing
end
JuMP.set_value(parameter::VariableRef, value::Float64; fix::Bool=true) = _set_value(parameter, value; fix=fix)

function enable_parameters!(model::JuMP.Model)
    haskey(model, :__parameters) && return nothing

    set_optimize_hook(model, _finalize_parameters)
    model.ext[:__parameters] = Dict{VariableRef, ParamData}()
    model.ext[:__links] = Dict{ConstraintRef, Vector{ParamVarLink}}()
    return nothing
end

_m = Model(HiGHS.Optimizer)
enable_parameters!(_m)

_x = @variable(_m, [1:2])
_p = @parameter(_m, [1:2], 1.0; fix=false)
# _c = @pconstraint(_m, _p .* _x .>= 1)
_c = @pconstraint(_m, [i=1:2], _p[i] * _x[i] >= 1.0)

@objective(_m, Min, sum(_x))
optimize!(_m)

_m.ext[:__links]
print(_m)

_m.ext[:__parameters][_p] = 0.0
optimize!(_m)




function build_and_pass(N::Int64, opt; direct=false, parametric=true)
    @info "Build model"
    @time begin
        if direct
            model = direct_model(opt())
        else
            model = Model(opt)
        end
        set_silent(model)
        set_string_names_on_creation(model, false)
        set_time_limit_sec(model, 1e6)

        enable_parameters!(model)

        @variable(model, x[i=1:N])
        if parametric
            p = @parameter(model, [i=1:N], 1.5; fix=false)
            c = @pconstraint(model, [i=1:N], p[i] * x[i] >= 1)
        else
            c = @constraint(model, [i=1:N], 1.5 * x[i] >= 1)
        end

        # @constraint(model, [i=1:N], x[i] >= 1. / i)

        @objective(model, Min, sum(x))
    end
    @info "First optimize"
    @time begin
        optimize!(model)
    end
    @info "Update"
    @time begin
        if parametric
            set_value.(p, 2.0; fix=false)
        else
            set_normalized_coefficient.(c, x, 2.0)
        end
    end
    @info "Re-optimize"
    @time optimize!(model)
    @info "Total"
    return model
end

@time build_and_pass(1, HiGHS.Optimizer);
@time build_and_pass(1, HiGHS.Optimizer; parametric=false);
@time build_and_pass(1, HiGHS.Optimizer; direct=true);

GC.gc()
@time build_and_pass(50_000, HiGHS.Optimizer; direct=true, parametric=false);
GC.gc()
@time build_and_pass(50_000, HiGHS.Optimizer; direct=true);


using ProfileView
ProfileView.@profview build_and_pass(50_000, HiGHS.Optimizer);

