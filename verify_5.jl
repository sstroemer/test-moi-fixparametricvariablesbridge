import HiGHS
import Gurobi
include("new_idea_5.jl")

# ======= 1 =========
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

# ======= 2 =========
model = JuMP.Model(HiGHS.Optimizer)
enable_parameters!(model)

@variable(model, x[1:3] >= 0)
@parameter(model, p[1:2], [1.0, 1.0]; fix=false)
@parameter(model, q, 5.0)

e1 = @pexpression(model, [i=1:2], x[i] + 1)

add_to_expression!(e1[1], x[2]*p[1])
add_to_expression!(e1[2], x[3]*p[2])

e2 = @pexpression(model, q + 5)

@pconstraint(model, c[i=1:2], e1[i] + e2 >= 10)

optimize!(model)
print(c[1])

set_value(p[1], 5.0; fix=false)
optimize!(model)
print(c[1])
