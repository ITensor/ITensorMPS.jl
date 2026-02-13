module ITensorMPSObserversExt
using ITensorMPS: ITensorMPS
using Observers.DataFrames: AbstractDataFrame
using Observers: Observers

function ITensorMPS.update_observer!(observer::AbstractDataFrame; kwargs...)
    return Observers.update!(observer; kwargs...)
end
end
