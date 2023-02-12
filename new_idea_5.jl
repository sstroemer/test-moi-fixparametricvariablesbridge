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
    constraint_index::ConstraintRef#MOI.ConstraintIndex
    variable_index::VariableRef#MOI.VariableIndex
    update::Update
end
UpdateLink = UpdateLink3

struct ParametricConstraint6{S} <: AbstractConstraint
    f::AffExpr
    s::S
    temporal_update_links::Dict{VariableRef, Update}#Vector{Tuple}
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
    for (vars, coeff) in terms
        if isnothing(model)
            model = owner_model(vars.a)
            n_param = length(model.ext[:__parameters])
        end

        v_a = get(model.ext[:__parameters], vars.a, nothing)
        v_b = get(model.ext[:__parameters], vars.b, nothing)

        if v_a !== nothing
            if !haskey(updates, vars.b)
                aff = get(affine.terms, vars.b, 0.0)
                updates[vars.b] = Update(aff, spzeros(n_param))
            end

            updates[vars.b].params[v_a.index] += coeff
        elseif v_b !== nothing
            if !haskey(updates, vars.a)
                aff = get(affine.terms, vars.a, 0.0)
                updates[vars.a] = Update(aff, spzeros(n_param))
            end

            updates[vars.a].params[v_b.index] += coeff
        else
            _error("no")
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

