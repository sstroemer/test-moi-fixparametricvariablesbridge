include("new_idea_5.jl")


model = JuMP.Model(HiGHS.Optimizer)
enable_parameters!(model)

@variable(model, x[1:3] >= 0)
@parameter(model, p[1:2], [1.0, 1.0]; fix=false)
@parameter(model, q, 5.0)

@pconstraint(model, c, (p[1] + p[2]) * x[1] + 2*p[2]*x[2] + 2*x[1] + x[3] >= q)

@objective(model, Min, sum(x))

optimize!(model)
print(c)
reduced_cost(q)

set_value(p[2], 0.0; fix=false)
print(c)

optimize!(model)
print(c)
reduced_cost(q)
