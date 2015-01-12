function read_svm_data(pathname::String)
	mat_x_i = Array(Int, 0)
	mat_x_j = Array(Int, 0)
	mat_x_v = Array(Float64, 0)
	vec_y = Array(Float64, 0)
	io = open(pathname, "r")
	j = 1
	for line in eachline(io)
		fields = split(chomp(line), " ")
		push!(vec_y, parsefloat(Float64, fields[1]))
		for k in 2:(length(fields) - 1)
			feature_index, feature_value = split(fields[k], ":")
			push!(mat_x_i, parseint(Int, feature_index))
			push!(mat_x_j, j)
			push!(mat_x_v, parsefloat(Float64, feature_value))
		end
		j = j + 1
	end
	close(io)
	return vec_y, sparse(mat_x_i, mat_x_j, mat_x_v)
end

