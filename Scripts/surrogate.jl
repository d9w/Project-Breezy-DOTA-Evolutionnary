using CSV
using DataFrames
using Cambrian
using Flux.Data: DataLoader
using Flux: @epochs
using CUDAapi
using BSON: @save, @load
if has_cuda()
    import CuArrays
    CuArrays.allowscalar(false)
end

function get_data(;log_file="all_episodes.csv", batch=1024)
    dat = CSV.File(log_file, header=false) |> DataFrame!
    X = Array{Float64}(dat[1:size(dat, 1)-1, 1:143])
    y = Array{Float64}(dat[:, 144:147])
    dy = diff(y, dims=1)
    dy[dy .< 0] .= 0.0
    dy[:, 1] ./= 10
    r = sum(dy, dims=2)
    test = rand(size(X, 1)) .< 0.2
    train_data = DataLoader(X[.~test, :]', r[.~test, :]', batchsize=batch, shuffle=true)
    test_data = DataLoader(X[test, :]', r[test, :]', batchsize=batch)
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
    train_data, test_data = get_data()
    train_data = device.(train_data)
    test_data = device.(test_data)
    m = device(m)
    loss(x, y) = Flux.mse(m(x), y)

    evalcb = () -> @show((loss_all(train_data, m), loss_all(test_data, m)))
    @epochs n_epochs Flux.train!(loss, params(m), train_data, ADAM(), cb=evalcb)
    if save
        @save "reward_model.bson" cpu(m)
    end
    m
end

function get_model()
    m = build_model()
    @load "reward_model.bson" m
    return m
end

"""append episode log to all episodes"""
function append_episode_logs(episode_log="episode_log.csv", all_episodes="all_episodes.csv")
    println("write_episode_logs")
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
        reward = model([inputs[:, 1]; onehot])[1]
        t_reward += reward
    end
    t_reward
end
