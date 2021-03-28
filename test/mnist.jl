using FluxProgress
using Flux
Flux.Data.MNIST
import Flux.Losses.logitcrossentropy
import Flux.@epochs
import Flux.onehotbatch
using Zygote

# Train dataset
imgs = Flux.Data.MNIST.images()
labels = Flux.Data.MNIST.labels()

dataset = (imgs, labels)
trainloader = Flux.Data.DataLoader(dataset, batchsize=500, shuffle=true)

nclass = 10
model = Chain(
    Dense(28*28, 128, relu),
    Dense(128, 32, relu),
    Dense(32, nclass),
)
model = gpu(model)

ps = params(model)
opt = ADAM()

"""
Image classifier
input x and y is cpu array
"""
function loss(x, y)
    x, y = convertinput(x, y)
    ŷ = model(x)
    l = logitcrossentropy(ŷ, y; dims=1)
    return l
end

function forward(X)
    x = reduce(hcat, [collect(Iterators.flatten(Float32.(x))) for x in X])
    x = gpu(x)
    return model(x)
end

function convertinput(X, y)
    x = reduce(hcat, [collect(Iterators.flatten(Float32.(x))) for x in X])
    x = gpu(x)
    y = gpu(Float32.(onehotbatch(y, 0:9)))
    return x, y
end
Zygote.@nograd convertinput

for i in 1:3
    FluxProgress.trainwithprogress!(loss, ps, trainloader, opt;)
    @show loss(imgs[1:4], labels[1:4])
end