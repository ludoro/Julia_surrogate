using XGBoost
using DifferentialEquations

# Parametrized Lotka-Volterra equations with coefficients:
function LotkaVolterra(du,u,p,t)
    du[1] = p[1] * u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end
u0 = [1.2; 2.4]
tspan = (0.0,3.0)

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

label = matrix_solutions[:,1]
dtrain = DMatrix(matrix_parameters, label=label)
num_round = 2
bst = xgboost(matrix_parameters, num_round, label = label, eta = 1, max_depth = 2)
test = rand(.5:0.1:4,4)
pred = predict(bst,test)
print(pred)
