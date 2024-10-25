using ITensors: ITensor, inds, permute

# Represents a time-dependent sum of terms:
#
# expr(t) = coefficients(expr)[1](t) * terms(expr)[1] + coefficients(expr)[2](t) * terms(expr)[2] + …
#
struct TimeDependentSum{Coefficients,Terms}
  coefficients::Coefficients
  terms::Terms
end

coefficients(expr::TimeDependentSum) = expr.coefficients
terms(expr::TimeDependentSum) = expr.terms
function Base.copy(expr::TimeDependentSum)
  return TimeDependentSum(coefficients(expr), copy.(terms(expr)))
end

function Base.:*(c::Number, expr::TimeDependentSum)
  scaled_coefficients = map(coefficient -> (t -> c * coefficient(t)), coefficients(expr))
  return TimeDependentSum(scaled_coefficients, terms(expr))
end
Base.:*(expr::TimeDependentSum, c::Number) = c * expr

# Evaluating a `TimeDependentSum` at a certain time
# returns a `ScaledSum` at that time.
function (expr::TimeDependentSum)(t::Number)
  coefficients_t = map(coefficient -> coefficient(t), coefficients(expr))
  return ScaledSum(coefficients_t, terms(expr))
end

# alternating_update inteface
function reduced_operator(operator::TimeDependentSum)
  return TimeDependentSum(coefficients(operator), reduced_operator.(terms(operator)))
end
function ITensorMPS.set_nsite!(operator::TimeDependentSum, nsite)
  foreach(t -> set_nsite!(t, nsite), terms(operator))
  return operator
end
function ITensorMPS.position!(operator::TimeDependentSum, state, position)
  foreach(t -> position!(t, state, position), terms(operator))
  return operator
end

# Represents the sum of scaled terms:
#
# H = coefficient[1] * H[1] + coefficient * H[2] + …
#
struct ScaledSum{Coefficients,Terms}
  coefficients::Coefficients
  terms::Terms
end

coefficients(expr::ScaledSum) = expr.coefficients
terms(expr::ScaledSum) = expr.terms

# Apply the scaled sum of terms:
#
# expr(x) = coefficients(expr)[1] * terms(expr)[1](x) + coefficients(expr)[2] * terms(expr)[2](x) + …
#
# onto x.
function scaledsum_apply(expr, x)
  return mapreduce(+, zip(coefficients(expr), terms(expr))) do coefficient_and_term
    coefficient, term = coefficient_and_term
    return coefficient * term(x)
  end
end
(expr::ScaledSum)(x) = scaledsum_apply(expr, x)
(expr::ScaledSum)(x::ITensor) = permute(scaledsum_apply(expr, x), inds(x))
