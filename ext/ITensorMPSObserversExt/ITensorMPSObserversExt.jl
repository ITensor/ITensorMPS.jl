module ITensorMPSObserversExt
using Observers: Observers
using Observers.DataFrames: AbstractDataFrame
using ITensorMPS: ITensorMPS

function ITensorMPS.update_observer!(observer::AbstractDataFrame; kwargs...)
    return Observers.update!(observer; kwargs...)
end
end
