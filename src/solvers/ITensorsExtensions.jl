module ITensorsExtensions

using ITensors: ITensor, inds
using NamedDimsArrays: unname

function to_vec(x::ITensor)
  function to_itensor(x_vec)
    return ITensor(x_vec, inds(x))
  end
  return vec(unname(x)), to_itensor
end

end
