# Randomization option slows down processing
# but improves quality of solution considerably
# Would be better to do randomization in place
function cddual(X::SparseOrFullMat,
	            Y::Vector;
	            C::Real = 1.0,
	            norm::Integer = 2,
	            randomized::Bool = true,
	            maxpasses::Integer = 1000,
	            eps::Real = 0.1)
	const PG_EPS = 1.0e-12

	# l: # of samples
	# n: # of features
	n, l = size(X)
	alpha = zeros(l)
	w = zeros(n)

	# Set U and D
	#  * L1-SVM: U = C, D[i] = 0
	#  * L2-SVM: U = Inf, D[i] = 1 / (2C)
	U = 0.0
	D = Array(Float64, l)
	if norm == 1
		U = C
		fill!(D, 0.0)
	elseif norm	== 2
		U = Inf
		fill!(D, 1.0 / (2.0 * C))
	else
		DomainError("Only L1-SVM and L2-SVM are supported")
	end

	# Set Qbar
	Qbar = Array(Float64, l)
	for i in 1:l
		Qbar[i] = D[i] + vecnorm(X[:, i], 2)^2
	end

	# Loop over examples
	converged = false
	pass = 0

	indices = randomized ? randperm(l) : [1:l]

	while !converged
		pass += 1
		if pass > maxpasses
			break
		end
		inv_ins = 0
		pg_max = -Inf
		pg_min = Inf

		if randomized
			shuffle!(indices)
		end

		# Process all observations
		for i in indices
			g = Y[i] * inner_prod(w, X[:, i]) - 1.0 + D[i] * alpha[i]

			if alpha[i] == 0.0
				pg = min(g, 0.0)
				if g <= 0.0
					inv_ins += 1
				end
			elseif alpha[i] == U
				pg = max(g, 0.0)
				if g >= 0.0
					inv_ins += 1
				end
			else
				pg = g
			end
			pg_max = max(pg_max, pg)
			pg_min = min(pg_min, pg)

			if abs(pg) > PG_EPS
				alphabar = alpha[i]
				alpha[i] = min(max(alpha[i] - g / Qbar[i], 0.0), U)
				add_vec!(w, X[:, i], e -> (alpha[i] - alphabar) * Y[i] * e)
			end
		end

		if (pg_max - pg_min <= eps) && (inv_ins == 0)
			converged = true
			break
		end
	end

	return SVMFit(w, pass, converged)
end
