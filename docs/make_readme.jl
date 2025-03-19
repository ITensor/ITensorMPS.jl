using Literate: Literate
using ITensorMPS: ITensorMPS

Literate.markdown(
  joinpath(pkgdir(ITensorMPS), "examples", "README.jl"),
  joinpath(pkgdir(ITensorMPS));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
