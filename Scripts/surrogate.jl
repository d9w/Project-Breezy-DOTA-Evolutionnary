using CSV
using DataFrames
using Cambrian
using Flux
using Flux.Data: DataLoader
using Flux: @epochs
using CUDAapi
using BSON: @save, @load
if has_cuda()
    import CuArrays
    using CUDAdrv
    CuArrays.allowscalar(false)
end

function get_data(;log_file="all_episodes.csv", lookahead=10)
    dat = CSV.File(log_file, header=false) |> DataFrame!
    X = Array{Float64}(dat[:, 1:143])
    y = Array{Float64}(dat[:, 144:147])
    y = hcat(y, X[:, [2, 26]])
    dy = diff(y, dims=1)
    dy[:, 1] ./= 10
    dy[:, 5] .*= 10
    dy[:, 6] .*= -10
    dt = diff(X[:, 46], dims=1)
    start = dt .< 0.0
    lookahead = 20
    r = zeros(size(dy, 1), lookahead)
    r[:, 1] = sum(dy, dims=2)
    for i in 1:(size(dy, 1) - lookahead)
        for j in 2:lookahead
            if start[i+j]
                break
            end
            r[i, j] = r[i+j, 1]
        end
    end
    for i in 2:lookahead
        r[:, i] .*= (1.0 - (i / lookahead))
    end
    r = sum(r, dims=2)
    X = X[1:size(r, 1), :]
    X = X[.~start, :]
    r = r[.~start, :]
    X, r
end

function get_data_loaders(X::Array{Float64}, y::Array{Float64}; batch=1024)
    test = rand(size(X, 1)) .< 0.2
    train_data = DataLoader(X[.~test, :]', y[.~test, :]', batchsize=batch, shuffle=true)
    test_data = DataLoader(X[test, :]', y[test, :]', batchsize=batch)
    train_data, test_data
end


function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += Flux.mse(model(x), y)
    end
    l/length(dataloader)
end

function build_model()
    Chain(Dense(143, 128, relu),
          Dense(128, 128, relu),
          Dense(128, 64, relu),
          Dense(64, 32, relu),
          Dense(32, 1, relu))
end

function train_model(;device=cpu, n_epochs=10, save=false)
    m = build_model()
    X, y = get_data()
    train_data, test_data = get_data_loaders(X, y)
    train_data = device.(train_data)
    test_data = device.(test_data)
    m = device(m)
    loss(x, y) = Flux.mse(m(x), y)

    evalcb = () -> @show((loss_all(train_data, m), loss_all(test_data, m)))
    @epochs n_epochs Flux.train!(loss, params(m), train_data, ADAM(), cb=evalcb)
    if save
        m = cpu(m)
        @save "reward_model.bson" m
    end
    m
end

function get_model()
    @load "reward_model_2.bson" cm2
    cm2
end

"""append episode log to all episodes"""
function append_episode_logs(episode_log="episode_log.csv", all_episodes="all_episodes.csv")
    println("append_episode_logs")
    io_e = open(episode_log, read=true)
    io_all = open(all_episodes, append=true)
    write(io_all, read(io_e, String))
    close(io_all)
    close(io_e)
end

"""truncate episode log"""
function clear_episode_log(episode_log="episode_log.csv")
    io_e = open(episode_log, "w+")
    write(io_e, "")
    close(io_e)
end

function get_surrogate_inputs(episode_log="episode_log.csv")
    dat = CSV.File(episode_log, header=false) |> DataFrame!
    Array{Float64}(Array{Float64}(dat[:, 1:113])')
end

"""use the model to estimate the reward for each action"""
function get_all_actions(model, inputs::Array{Float64}; n_action=30)
    rewards = zeros(Float32, n_actions)
    for i in eachindex(rewards)
        actions = 0:29 .== i
        rewards[i] = model([inputs; actions])[1]
    end
    rewards
end

"""returns the total reward of an individual based on logged states"""
function simulate(ind::Individual, model, inputs::Array{Float64})
    t_reward = 0.0
    for i in 1:size(inputs, 2)
        outputs = process(ind, inputs[:, i])
        onehot = eachindex(outputs) .== argmax(outputs)
        reward = model([inputs[:, i]; onehot])[1]
        t_reward += reward
    end
    t_reward
end
