module ITensorsExtensions
using ITensors: ITensor, array, inds, itensor
function to_vec(x::ITensor)
    function to_itensor(x_vec)
        return itensor(x_vec, inds(x))
    end
    return vec(array(x)), to_itensor
end
end
