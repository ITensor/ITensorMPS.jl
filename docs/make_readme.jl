using Literate: Literate
using ITensorMPS: ITensorMPS

function ccq_logo(content)
  include_ccq_logo = """
    <img src="docs/src/assets/CCQ.png" width="20%" alt="Flatiron Center for Computational Quantum Physics logo.">
    """
  content = replace(content, "{CCQ_LOGO}" => include_ccq_logo)
  return content
end

Literate.markdown(
  joinpath(pkgdir(ITensorMPS), "examples", "README.jl"),
  joinpath(pkgdir(ITensorMPS));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
  postprocess=ccq_logo,
)
