using DifferentialEquations
using GLM
using Distributions
using DataFrames
using GaussianProcesses
using Stheno
# Parametrized Lotka-Volterra equations with coefficients:
function LotkaVolterra(du,u,p,t)
    du[1] = p[1] * u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] +p[4]*u[1]*u[2]
end
u0 = [1.2; 2.4]
tspan = (0.0,3.0)
#p = [1.0,1.5,1.2,1.3]


# 10 different sets of p
# predicting the solution(3.0) = x,y_final
matrix_parameters = Array{Float64}(undef, 10, 4)
solution_tfinal = Array{Float64}(undef, 10,2)

for i=1:10
    p = rand(.5:0.1:4,4)
    matrix_parameters[i,1:4] = p
    prob = ODEProblem(LotkaVolterra,u0,tspan,p)
    solution = solve(prob,Tsit5(),save_everystep=false)
    solution_tfinal[i,1] =solution(3.0)[1]
    solution_tfinal[i,2] =solution(3.0)[2]
end
