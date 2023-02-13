using JuMP
using OrderedCollections
using SparseArrays

model = Model(HiGHS.Optimizer)

struct Test1
    a
    b
end

e = GenericAffExpr{Test1,VariableRef}(Test1(0.0, 0.0))

Base.copy(x::Test1) = x
Base.one(::Type{Test1}) = Test1(1.0, spzeros(100000))
_constant_to_number(x::Test1) = x

x = @variable(model)

add_to_expression!(e, Test1(1.0, spzeros(100000)), x)

function add_to_expression!(
    aff::GenericAffExpr{Test1,V},
    new_coef::Test1,
    new_var::V,
) where {V}
    _add_or_set!(aff.terms, new_var, convert(Test1, _constant_to_number(new_coef)))
    return aff
end

function _add_or_set!(dict::OrderedDict{V,Test1}, k::V, v::Test1) where {V}
    # Adding zero terms to this dictionary leads to unacceptable performance
    # degradations. See, e.g., https://github.com/jump-dev/JuMP.jl/issues/1946.
#    if iszero(v)
#        return dict  # No-op.
#    end
    index = OrderedCollections.ht_keyindex2(dict, k)
    if index <= 0  # Key does not exist. We pay the penalty of a second lookup.
        setindex!(dict, v, k)
    else
        dict.vals[index] = Test1(dict.vals[index].a + v.a, spzeros(10000))
        dict.keys[index] = k
    end
    return dict
end
