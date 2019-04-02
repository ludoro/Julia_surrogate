using GaussianProcesses
using DifferentialEquations


# ReParametrized Lokta to have just one coefficient as seen here:
# https://www.maths.dur.ac.uk/~ktch24/term1Notes(10).pdf


function LotkaVolterraChanged(du,u,p,t)
    du[1] = u[1] - u[1]*u[2]
    du[2] = +p*(-u[2] + u[1]*u[2])
end
u0 = [1.2; 2.4]
tspan = (0.0,3.0)

p = rand(.5:0.1:4,10)
matrix_solutions = Array{Float64}(undef, 10,2)
for i = 1:10
    prob = ODEProblem(LotkaVolterraChanged,u0,tspan,p[i])
    solution = solve(prob,Tsit5(),save_everystep=false)
    matrix_solutions[i,1] = solution(3.0)[1]
    matrix_solutions[i,2] = solution(3.0)[2]
end
println(matrix_solutions[:,1])

kern = SE(0.0,0.0)
mean = MeanZero()
gp = GP(p,matrix_solutions[:,1],mean,kern)
optimize!(gp)

p_test = rand(.5:0.1:4)
#Mu contains the solution
mu,sigma = predict_y(gp,p_test)



# Input a vector of 4 parameters, still single output, I get error saying that input and outpust must have same dimensions
#(no known library that does multi output GP's)
#=
function LotkaVolterra(du,u,p,t)
    du[1] = p[1] * u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end
u0 = [1.2; 2.4]
tspan = (0.0,3.0)

matrix_parameters = Array{Float64}(undef,10, 4)
matrix_solutions = Array{Float64}(undef, 10,2)
for i = 1:10
    matrix_parameters[i,:] = rand(.5:0.1:4,4)
    prob = ODEProblem(LotkaVolterra,u0,tspan,matrix_parameters[i,:])
    solution = solve(prob,Tsit5(),save_everystep=false)
    matrix_solutions[i,1] = solution(3.0)[1]
    matrix_solutions[i,2] = solution(3.0)[2]
end

kern = SE(0.0,0.0)
mean = MeanZero()
gp = GP(matrix_parameters,matrix_solutions[:,1], mean, kern)
optimize!(gp)

p_test = rand(0.5:0.1:4,4)
mu,sigma = predict_y(gp,p_test)
#=
