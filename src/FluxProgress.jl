module FluxProgress

export trainwithprogress!

using ProgressMeter
import Flux.train!
import Zygote: Params, gradient
import Flux.Optimise:update!, runall, batchmemaybe, StopException, SkipException


"""
trainwithprogress!
"""
function trainwithprogress!(loss, ps, data, opt; cb=() -> ())
    p = Progress(
        length(data),
        dt=0.0,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
    )
    ps = Params(ps)
    cb = runall(cb)
    for (index, d) in enumerate(data)
        nbatch = length(d[1])
        try
        local l
        gs = gradient(ps) do
            l = loss(batchmemaybe(d)...)
        end
        update!(opt, ps, gs)
        cb()
        next!(
            p;
            showvalues=[
            (:index, index),
            (:loss, l/nbatch),
            ]
        )
    catch ex
        if ex isa StopException
            break
        elseif ex isa SkipException
            continue
        else
            rethrow(ex)
        end
      end
    end
end

end # module



