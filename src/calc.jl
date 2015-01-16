function add_vec!(w::DenseVector, x::DenseVector, f::Function)
	if length(w) != length(x)
		throw(DimensionMismatch())
	end
	for i in 1:length(x)
		w[i] += f(x[i])
	end
end

function add_vec!(w::DenseVector, X::SparseMatrixCSC, f::Function)
	n, l = size(X)
	# Assumes that X is a column vector
	if length(w) != n || l != 1
		throw(DimensionMismatch())
	end
	I, J, V = findnz(X)
	for k in 1:length(I)
		w[I[k]] += f(V[k])
	end
end

function inner_prod{T<:Real}(x::DenseVector{T}, y::DenseVector{T})
	dot(x, y)
end

function inner_prod{T<:Real}(X::SparseMatrixCSC{T}, y::DenseVector{T})
	inner_product(y, X)
end

function inner_prod{T<:Real}(x::DenseVector{T}, Y::SparseMatrixCSC{T})
	n, l = size(Y)
	# Assumes that X is a column vector
	if length(x) != n || l != 1
		throw(DimensionMismatch())
	end
	if n == 0
		return zero(eltype(x))*zero(eltype(Y))
	end
	I, J, V = findnz(Y)
	if size(I) == 0
		return zero(eltype(x))*zero(eltype(Y))
	end
	s = x[I[1]] * V[1]
	@inbounds for k in 2:length(I)
		s += x[I[k]] * V[k]
	end
	s
end

