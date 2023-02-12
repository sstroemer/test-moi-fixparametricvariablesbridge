using SparseArrays
import LinearAlgebra
import HiGHS
import Gurobi
using JuMP
using OrderedCollections


sv = spzeros(3)

function mmdot(x, y)
    lx = length(x)
    ly = length(y)

    if lx == ly
        return LinearAlgebra.dot(x, y)
    elseif lx < ly
        return LinearAlgebra.dot(x, @view(y[1:lx]))
    else
        return LinearAlgebra.dot(@view(x[1:ly]), y)
    end
end

struct Update1
    constant::Float64
    params::SparseVector{Float64, UInt64}
end
Update = Update1

struct UpdateLink3
    constraint_index::ConstraintRef
    variable_index::VariableRef
    update::Update
end
UpdateLink = UpdateLink3

struct ParametricConstraint6{S} <: AbstractConstraint
    f::AffExpr
    s::S
    temporal_update_links::Dict{VariableRef, Update}
end
ParametricConstraint{S} = ParametricConstraint6{S}

mutable struct ParamData2
    index::UInt64
    value::Float64
end
ParamData = ParamData2

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

    model = nothing # todo
    n_param = 0  

    updates = Dict{VariableRef, Update}()
    sizehint!(updates, length(terms))
    for (vars, coeff) in terms
        if isnothing(model)
            model = owner_model(vars.a)
            n_param = length(model.ext[:__parameters])
        end

        v_a = get(model.ext[:__parameters], vars.a, nothing)
        v_b = get(model.ext[:__parameters], vars.b, nothing)

        if v_a !== nothing
            if !haskey(updates, vars.b)
                updates[vars.b] = Update(get(affine.terms, vars.b, 0.0), spzeros(n_param))
            end

            updates[vars.b].params[v_a.index] += coeff
        elseif v_b !== nothing
            if !haskey(updates, vars.a)
                updates[vars.a] = Update(get(affine.terms, vars.a, 0.0), spzeros(n_param))
            end

            updates[vars.a].params[v_b.index] += coeff
        else
            _error("At least one parameter is not properly registered.")
        end
    end

    new_set = typeof(set)(MOI.constant(set) - affine.constant)
    affine.constant = 0

    return ParametricConstraint(affine, new_set, updates)
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

    for (k, v) in con.temporal_update_links
        push!(model.ext[:__links], UpdateLink(constr, k, v))
    end

    return constr
end

function eval_var_coeff(update::Update, param_values::Vector{Float64})
    return update.constant + mmdot(update.params, param_values)
end


function _finalize_parameters(model::JuMP.Model)
    # be = backend(model)

    param_values = collect(it[2].value for it in model.ext[:__parameters])

    @inbounds @simd for ul in model.ext[:__links]
        # somehow this results in less allocations but more time:
        # MOI.modify(
        #     be, ul.constraint_index,
        #     MOI.ScalarCoefficientChange(ul.variable_index, eval_var_coeff(model, ul.update))
        # )
        set_normalized_coefficient(ul.constraint_index, ul.variable_index, eval_var_coeff(ul.update, param_values))
    end
    # model.is_model_dirty = true

    return optimize!(model; ignore_optimize_hook = true)
end

function enable_parameters!(model::JuMP.Model)
    haskey(model, :__parameters) && return nothing

    set_optimize_hook(model, _finalize_parameters)
    model.ext[:__parameters] = OrderedDict{VariableRef, ParamData}()
    model.ext[:__links] = Vector{UpdateLink}()
    return nothing
end

function _set_value(parameter::VariableRef, value::Float64; fix::Bool=true)
    model = owner_model(parameter)
    if !haskey(model.ext[:__parameters], parameter)
        error("nope!")
    end

    model.ext[:__parameters][parameter].value = value
    fix && JuMP.fix(parameter, value; force=true)
    return nothing
end
JuMP.set_value(parameter::VariableRef, value::Float64; fix::Bool=true) = _set_value(parameter, value; fix=fix)

macro pexpression(args...)
    return esc(:($JuMP.QuadExpr.($JuMP.@expression($(args...)))))
end

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
    # if !(flat_args[end] isa Number) and !(flat_args[end] isa Vector)
    #     _error("Wrong arguments. Did you miss the initial parameter value?")
    # end
    _vector_scalar_get(_scalar::Number, ::Int64) = _scalar
    _vector_scalar_get(_vector::Vector{T} where T<:Number, i::Int64) = _vector[i]

    if get(kw_args, :fix, true)
        return esc(quote
            _var = $JuMP.@variable($(flat_args[1:(end-1)]...))
            if _var isa Vector
                for i in eachindex(_var)
                    $(flat_args[1]).ext[:__parameters][_var[i]] = ParamData(UInt64(length($(flat_args[1]).ext[:__parameters]) + 1), $_vector_scalar_get($(flat_args[end]), i))
                end
            else
                $(flat_args[1]).ext[:__parameters][_var] = ParamData(UInt64(length($(flat_args[1]).ext[:__parameters]) + 1), $(flat_args[end]))
                $JuMP.fix(_var, $(flat_args[end]); force=true)
            end
            _var
        end)
    else
        return esc(quote
            _var = $JuMP.@variable($(flat_args[1:(end-1)]...))
            if _var isa Vector
                for i in eachindex(_var)
                    $(flat_args[1]).ext[:__parameters][_var[i]] = ParamData(UInt64(length($(flat_args[1]).ext[:__parameters]) + 1), $_vector_scalar_get($(flat_args[end]), i))
                end
            else
                $(flat_args[1]).ext[:__parameters][_var] = ParamData(UInt64(length($(flat_args[1]).ext[:__parameters]) + 1), $(flat_args[end]))
            end
            _var
        end)
    end
end

