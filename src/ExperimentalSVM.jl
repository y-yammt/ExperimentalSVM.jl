module ExperimentalSVM

import StatsBase.predict

export svm, cddual, cddual_shrinking, pegasos, predict
export read_svm_data

typealias SparseOrFullMat Union(Matrix, SparseMatrixCSC)

type SVMFit
	w::Vector{Float64}
	pass::Int
	converged::Bool
end

function Base.show(io::IO, fit::SVMFit)
	@printf io "Fitted linear SVM\n"
	@printf io " * Non-zero weights: %d\n" countnz(fit.w)
	@printf io " * Iterations: %d\n" fit.pass
	@printf io " * Converged: %s\n" string(fit.converged)
end

function predict(fit::SVMFit, X::SparseOrFullMat, ret_class::Bool=true)
	n, l = size(X)
	preds = Array(Float64, l)
	for i in 1:l
		v = inner_prod(fit.w, X[:,i])
		preds[i] = ret_class ? sign(v) : v
	end
	return preds
end

include("calc.jl")

include("io.jl")

include("pegasos.jl")

include("cddual.jl")

include("cddual_shrinking.jl")

function svm(X::SparseOrFullMat,
	         Y::Vector;
	         k::Integer = 5,
	         lambda::Real = 0.1,
	         T::Integer = 100)
	pegasos(X, Y, k = k, lambda = lambda, T = T)
end

end # module
