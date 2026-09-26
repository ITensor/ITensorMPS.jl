using HDF5: HDF5, attributes, create_group, open_group, read, read_attribute, write
using ITensorMPS: MPS
using ITensors: ITensor

function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, M::MPS)
    closing(create_group(parent, name)) do g
        attributes(g)["type"] = "MPS"
        attributes(g)["version"] = 1
        N = length(M)
        write(g, "length", N)
        write(g, "rlim", M.rlim)
        write(g, "llim", M.llim)
        for n in 1:N
            write(g, "MPS[$(n)]", M[n])
        end
    end
    return
end

function HDF5.read(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{MPS})
    return closing(open_group(parent, name)) do g
        if read_attribute(g, "type") != "MPS"
            error("HDF5 group or file does not contain MPS data")
        end
        N = read(g, "length")
        rlim = read(g, "rlim")
        llim = read(g, "llim")
        v = [read(g, "MPS[$(i)]", ITensor) for i in 1:N]
        return MPS(v, llim, rlim)
    end
end
