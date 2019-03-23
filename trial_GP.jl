matrix_parameters = Array{Float64}(undef, 35, 5)
coll = collect(0.0:0.5:3.0)

for i = 1:35
    p = rand(.5:0.1:4,4)
    matrix_parameters[i,1:4] = p
end

coll2 = collect(1:7:35)
for i=1:length(coll2)
    matrix_parameters[coll2[i]:coll2[i]+6,5] = coll
end

println(matrix_parameters)
