using Compat: Returns

default_nsweeps() = nothing
default_checkdone() = Returns(false)
default_write_when_maxdim_exceeds() = nothing
default_nsite() = 2
default_reverse_step() = false
default_time_start() = nothing
default_time_step() = nothing
default_order() = 2
default_observer() = EmptyObserver()
default_sweep_observer() = EmptyObserver()
default_outputlevel() = 0
default_normalize() = false
default_sweep() = 1
