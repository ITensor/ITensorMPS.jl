module ITensorMPSHDF5Ext

# HDF5.jl has no do-block form of `open_group` or `create_group`, so without this
# the handles opened below stay alive until their finalizers run.
function closing(f, obj)
    try
        return f(obj)
    finally
        close(obj)
    end
end

include("mps.jl")
include("mpo.jl")
end
