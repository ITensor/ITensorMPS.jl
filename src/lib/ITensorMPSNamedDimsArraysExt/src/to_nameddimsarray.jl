using ..ITensorMPS: AbstractMPS
using ITensors.ITensorsNamedDimsArraysExt: ITensorsNamedDimsArraysExt, to_nameddimsarray

function ITensorsNamedDimsArraysExt.to_nameddimsarray(x::AbstractMPS)
  return to_nameddimsarray.(x)
end
