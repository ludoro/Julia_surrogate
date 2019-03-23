using DifferentialEquations
using GLM
using Distributions
using DataFrames
using GaussianProcesses

# Parametrized Lotka-Volterra equations with coefficients:
function LotkaVolterra(du,u,p,t)
    du[1] = p[1] * u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end
u0 = [1.2; 2.4]
tspan = (0.0,3.0)
#p = [1.0,1.5,1.2,1.3]
x =  Array{Float64}(undef, 0)
y =  Array{Float64}(undef, 0)
matrix_parameters = Array{Float64}(undef, 10, 4)
matrix_solutions = Array{Float64}(undef, 10,2)
for i = 1:10
    p = rand(.5:0.1:4,4)
    matrix_parameters[i,:] = p;
    prob = ODEProblem(LotkaVolterra,u0,tspan,p)
    solution = solve(prob,Tsit5(),save_everystep=false)
    matrix_solutions[i,1] = solution(3.0)[1]
    matrix_solutions[i,2] = solution(3.0)[2]
end

df_1 = DataFrame(matrix_parameters)
names!(df_1,[:p_1,:p_2,:p_3,:p_4,])
df_1[:x_end] = matrix_solutions[:,1]

df_2 = DataFrame(matrix_parameters)
names!(df_2,[:p_1,:p_2,:p_3,:p_4,])
df_2[:y_end] = matrix_solutions[:,2]

x_end_approxiamtion = lm(@formula(x_end ~ p_1 + p_2 + p_3 + p_4), df_1)
y_end_approxiamtion = lm(@formula(y_end ~ p_1 + p_2 + p_3 + p_4), df_2)

coef_x = coeftable(x_end_approxiamtion).cols[1]
coef_y = coeftable(y_end_approxiamtion).cols[1]

approximation = Array{Float64}(undef,2)
true_solution = Array{Float64}(undef,2)
a = 2.0;
b = 3.0;
c = 3.5;
d = 4.0;
approximation[1] = coef_x[1] + a*coef_x[2] + b*coef_x[3] + c*coef_x[4]
approximation[2] = coef_y[1] + a*coef_y[2] + b*coef_y[3] + c*coef_y[4]
p_trial = [a,b,c,d]
problem = ODEProblem(LotkaVolterra,u0,tspan,p_trial)
solution = solve(problem,Tsit5(),save_everystep=false)
true_solution[1] = solution(3.0)[1]
true_solution[2] = solution(3.0)[2]

println("The true solution is:")
println(true_solution[1])
println(true_solution[2])
println("The approximated solution is:")
println(approximation[1])
println(approximation[2])
