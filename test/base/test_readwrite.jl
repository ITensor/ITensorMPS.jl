@eval module $(gensym())
using HDF5
using ITensorMPS
using ITensors
using Test

include(joinpath(@__DIR__, "utils", "util.jl"))

@testset "HDF5 Read and Write" begin
    @testset "MPO/MPS" begin
        N = 6
        sites = siteinds("S=1/2", N)

        # MPO
        mpo = makeRandomMPO(sites)

        h5open("data.h5", "w") do fo
            return write(fo, "mpo", mpo)
        end

        h5open("data.h5", "r") do fi
            rmpo = read(fi, "mpo", MPO)
            @test prod([norm(rmpo[i] - mpo[i]) / norm(mpo[i]) < 1.0e-10 for i in 1:N])
        end

        # MPS
        mps = makeRandomMPS(sites)
        h5open("data.h5", "w") do fo
            return write(fo, "mps", mps)
        end

        h5open("data.h5", "r") do fi
            rmps = read(fi, "mps", MPS)
            @test prod([norm(rmps[i] - mps[i]) / norm(mps[i]) < 1.0e-10 for i in 1:N])
        end
    end

    @testset "No leaked HDF5 handles" begin
        nopen(file) = HDF5.API.h5f_get_obj_count(file, HDF5.API.H5F_OBJ_ALL)
        sites = siteinds("S=1/2", 20)
        mps = makeRandomMPS(sites)
        mpo = makeRandomMPO(sites)
        GC.gc()
        h5open("data.h5", "w") do fo
            write(fo, "mps", mps)
            write(fo, "mpo", mpo)
            # `h5f_get_obj_count` includes the file handle itself.
            @test nopen(fo) == 1
        end
        h5open("data.h5", "r") do fi
            read(fi, "mps", MPS)
            read(fi, "mpo", MPO)
            @test nopen(fi) == 1
        end
    end

    #
    # Clean up the test hdf5 file
    #
    rm("data.h5"; force = true)
end
end
