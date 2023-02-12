include("new_idea_5.jl")


model = JuMP.Model(HiGHS.Optimizer)
enable_parameters!(model)

@variable(model, x[1:3] >= 0)
@parameter(model, p[1:2], [1.0, 1.0]; fix=false)

@pconstraint(model, c1, (p[1] + p[2]) * x[1] + 2*p[2]*x[2] + 2*x[1] + x[3] >= 5)

@objective(model, Min, sum(x))

optimize!(model)
print(c1)

set_value(p[2], 0.0; fix=false)
# model.ext[:__parameters][p[2]].value = 0.0
print(c1)

optimize!(model)
print(c1)
