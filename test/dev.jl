using FluxProgress
using Flux

model = Flux.Dense(3, 3)
loss(x) = sum(abs.(model(x))
ps = params(model)
opt = ADAM()
data = [rand(3, 2) for _ in 1:300]

FluxProgress.trainwithprogress!(loss, ps, data, opt;)