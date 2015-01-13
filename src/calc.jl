function add_vec!(w::DenseVector, x::DenseVector, f::Function)
	if length(w) != length(x)
		throw(DimensionMismatch("Inconsistent dimensions."))
	end
	for i in 1:length(x)
		w[i] += f(x[i])
	end
end

function add_vec!(w::DenseVector, X::SparseMatrixCSC, f::Function)
	n, l = size(X)
	if length(w) != n || l != 1
		throw(DimensionMismatch("Inconsistent dimensions."))
	end
	I, J, V = findnz(X)
	for k in 1:length(I)
		w[I[k]] += f(V[k])
	end
end

