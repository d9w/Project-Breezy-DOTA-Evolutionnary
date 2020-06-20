using CSV
using DataFrames
using DecisionTree
using Cambrian

function get_data(log_file="all_episodes.csv")
    dat = CSV.File(log_file, header=false) |> DataFrame!
    X = dat[1:(size(dat, 1)-1), 1:143]
    y = dat[2:end, 144:147]
    Array{Float64}(X), Array{Float64}(y)
end

function train_model(X::Array{Float64}, y::Array{Float64})
    rf = RandomForestRegressor(n_trees=40)
    fit!(rf, X, y)
    rf
end

"""get prediction models for each output feature"""
function train_surrogate_models()
    X, y = get_data()
    models = []
    for i in 1:size(y, 2)
        m = train_model(X, y[:, i])
        push!(models, m)
    end
    models
end

"""append episode log to all episodes"""
function write_episode_logs(episode_log="episode_log.csv", all_episodes="all_episodes.csv")
    io_e = open(episode_log, read=true)
    io_all = open(all_episodes, append=true)
    write(io_all, read(io_e, String))
    close(io_all)
    close(io_e)
    io_e = open(episode_log, "w+")
    write(io_e, "")
    close(io_e)
end

"""reward based on net worth, last hits, denies, and tower damage predictions"""
function predict_reward(models, inputs::Array{Float64})
    net_worth = 0.1 * predict(models[1], inputs)
    last_hits = predict(models[2], inputs)
    denies = predict(models[3], inputs)
    ratio_tower = predict(models[4], inputs)
	  net_worth + 100*last_hits + 100*denies + 2000*ratio_tower
end

function get_surrogate_inputs(episode_log="episode_log.csv")
    dat = CSV.File(episode_log, header=false) |> DataFrame!
    Array{Float64}(Array{Float64}(dat[:, 1:113])')
end

"""returns the total reward of an individual based on logged states"""
function simulate(ind::Individual, models, inputs::Array{Float64})
    t_reward = 0.0
    for i in 1:size(inputs, 2)
        outputs = process(ind, inputs[:, i])
        onehot = eachindex(outputs) .== argmax(outputs)
        reward = predict_reward(models, [inputs[:, 1]; onehot])
        t_reward += reward
    end
    t_reward
end
