function cddual_shrinking(X::SparseOrFullMat,
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
		for i in 1:l
			D[i] = 0.0
		end
	elseif norm	== 2
		U = Inf
		for i in 1:l
			D[i] = 1.0 / (2.0 * C)
		end
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

	pg_max_old = Inf
	pg_min_old = -Inf
	active_size = l

	indices = randomized ? randperm(l) : [1:l]

	while !converged
		pass += 1
		if pass > maxpasses
			break
		end

		pg_max_new = -Inf
		pg_min_new = Inf

		if randomized && active_size > 1
			for i = active_size:-1:2
				j = rand(1:i)
				indices[i], indices[j] = indices[j], indices[i]
			end
		end

		k = 1
		while k <= active_size
			decrement_active = false

			i = indices[k]
			g = Y[i] * inner_prod(w, X[:, i]) - 1.0 + D[i] * alpha[i]

			if alpha[i] == 0.0
				if g > pg_max_old
					decrement_active = true
				end
				pg = min(g, 0.0)
			elseif alpha[i] == U
				if g < pg_min_old
					decrement_active = true
				end
				pg = max(g, 0.0)
			else
				pg = g
			end

			pg_max_new = max(pg_max_new, pg)
			pg_min_new = min(pg_min_new, pg)

			if abs(pg) > PG_EPS
				alphabar = alpha[i]
				alpha[i] = min(max(alpha[i] - g / Qbar[i], 0.0), U)
				add_vec!(w, X[:, i], e -> (alpha[i] - alphabar) * Y[i] * e)
			end

			if decrement_active
				indices[k], indices[active_size] = indices[active_size], indices[k]
				active_size -= 1
			else
				k += 1
			end
		end

		if (pg_max_new - pg_min_new) <= eps
			if active_size == l
				converged = true
				break
			else
				active_size = l
				pg_max_old = Inf
				pg_min_old = -Inf
			end
		else
			pg_max_old = pg_max_new > 0.0 ? pg_max_new : Inf
			pg_min_old = pg_min_new < 0.0 ? pg_min_new : -Inf
		end
	end
	return SVMFit(w, pass, converged)
end
