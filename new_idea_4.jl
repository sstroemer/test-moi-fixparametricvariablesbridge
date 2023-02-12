using JuMP
import HiGHS

using DynamicExpressions
operators = OperatorEnum(; binary_operators=[+, *])


struct UpdateLink1
    constraint_index::MOI.ConstraintIndex
    variable_index::MOI.VariableIndex
    e::Node{Float64}
end
UpdateLink = UpdateLink1

struct ParametricConstraint5{S} <: AbstractConstraint
    f::AffExpr
    s::S
    temporal_update_links::Vector{Tuple}
end
ParametricConstraint{S} = ParametricConstraint5{S}

struct ParamData1
    index::UInt64
end
ParamData = ParamData1

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
      
    expressions = Dict{VariableRef, Node{Float64}}()
    for (vars, coeff) in terms
        if isnothing(model)
            model = owner_model(vars.a)
        end

        v_a = get(model.ext[:__parameters], vars.a, nothing)
        v_b = get(model.ext[:__parameters], vars.b, nothing)

        if v_a !== nothing
            if !haskey(expressions, vars.b)
                aff = get(affine.terms, vars.b, 0.0)
                expressions[vars.b] = Node(Float64; val=aff)
            end

            expressions[vars.b] += coeff * model.ext[:__nodes][v_a.index]
        elseif v_b !== nothing
            if !haskey(expressions, vars.a)
                aff = get(affine.terms, vars.a, 0.0)
                expressions[vars.a] = Node(Float64; val=aff)
            end

            expressions[vars.a] += coeff * model.ext[:__nodes][v_b.index]
        else
            _error("no")
        end
    end

    temporal_update_links = Tuple[]
    sizehint!(temporal_update_links, length(expressions))
    for (k, v) in expressions
        push!(temporal_update_links, (k.index, v))
        # affine.terms[k] = f(model)
    end

    new_set = typeof(set)(MOI.constant(set) - affine.constant)
    affine.constant = 0

    return ParametricConstraint(affine, new_set, temporal_update_links)
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

    for tul in con.temporal_update_links
        push!(model.ext[:__links], UpdateLink(constr.index, tul[1], tul[2]))
    end

    return constr
end


function _finalize_parameters(model::JuMP.Model)
    arr = reshape(model.ext[:__parameter_values], length(model.ext[:__parameter_values]), 1)
    be = backend(model)

    for ul in model.ext[:__links]
        MOI.modify(
            be, ul.constraint_index,
            MOI.ScalarCoefficientChange(ul.variable_index, ul.e(arr)[1])
        )
    end
    model.is_model_dirty = true

    return optimize!(model; ignore_optimize_hook = true)
end

function enable_parameters!(model::JuMP.Model)
    haskey(model, :__parameters) && return nothing

    set_optimize_hook(model, _finalize_parameters)
    model.ext[:__parameters] = Dict{VariableRef, ParamData}()
    model.ext[:__nodes] = Vector{DynamicExpressions.Node{Float64}}()
    model.ext[:__links] = Vector{UpdateLink}()
    model.ext[:__parameter_values] = Vector{Float64}()
    return nothing
end

