using PyCall
using Pkg
using Cambrian
using Distances

## Import Simulator
pushfirst!(PyVector(pyimport("sys")."path"), "C:\\Users\\denni\\Documents\\GitHub\\Dota_Simulator")
dotasimlib = pyimport("DOTA_simulator")

"""
This code is not of my belonging but is from the https://github.com/TemplierPaul/Dota_Simulator .
It is just convenient to have it running from the Scripts folder
"""
function train_sim(epochs=30, batch_size=256, limit_overfit=10, name::String ="julia_cpu_test")
    # import DOTA_simulator as dotasimlib
    sim = dotasimlib.DotaSim()
    sim.load_data("../Dota_Simulator/games_data/rd_big*", true, true)
    print(sim)
    sim.set_model("ffnet",nothing,nothing,2,4000)
    sim.train(epochs=epochs, batch_size=epochs, limit_overfit=limit_overfit, plot=false)
    sim.save_model(name)
    sim
end

function import_sim(name="julia_cpu_test")
    # Create simulator
    sim = dotasimlib.DotaSim()
    println("----- IMPORTING SIMULATOR -----\n")
    sim.set_model(name=name)
    sim.model.use_cuda = false
    sim
end

function run_steps(sim, indiv::Individual, n_steps=100, render=true)
    state = sim.reset()
    if render
        sim.render()
    end
    actions::Array{Integer}=[]
    for i in 1:n_steps
        inputs = get_inputs(state)
        a = argmax(process(indiv, inputs)) - 1 #action in (0, 29)
        actions = append!(actions, [a])
        state = sim.step(a)
        if render
            sim.render()
        end
    end
    actions
end

# ## TEST TOOLS
# mutable struct testIndiv <: Individual
#     genes::Array{Float64}
#     fitness::Array{Float64}
# end

# function random_test_indiv()
#     testIndiv(rand(2), [0.])
# end

# function process(indiv::Individual, last_features::Array{Float64}=[0.])
#     a = zeros(30)
#     a[rand(1:30)] = 1
#     a
# end

## Indiv + distances to others
mutable struct Indiv_dist
    indiv::Individual
    distances::Array{Float64}
    actions::Array{Integer}
end

function set_struct(indiv::Individual, indiv_nb::Integer, sim)
    actions_list = run_steps(sim, indiv, 500, true)
    Indiv_dist(indiv, zeros(indiv_nb), actions_list)
end

function compute_distances(indivs::Array{Indiv_dist})
    size = length(indivs)
    len = length(indivs[1].actions)
    for i in 1:size
        for j in 1:i
            if i != j
                indiv_i = indivs[i]
                indiv_j = indivs[j]
                distance = sum(indiv_i.actions .!= indiv_j.actions)/len
                indiv_i.distances[j] = distance
                indiv_j.distances[i] = distance
            end
        end
    end
end

function select_diverse(indivs::Array{Individual, 1}, keep_top=10)
    # Import Simulator
    sim = import_sim()

    # Create Indiv_dist objects and compute 100 actions
    pop::Array{Indiv_dist, 1} = []
    for i in indivs
        pop = append!(pop, [set_struct(i, length(indivs), sim)])
    end

    # println("Group size: ", length(pop))

    compute_distances(pop)
    sort!(pop, by= i -> sum(i.distances), rev=true)
    println("Distances: ")
    println([sum(pop[i].distances) for i in 1:keep_top])

    pop = pop[1:keep_top]
    new_pop = getfield.(pop, :indiv)
    new_pop
end

## Test
function test()
    pop::Array{Individual, 1} = []
    for i in 1:20
        pop = append!(pop, [random_test_indiv()])
    end
    pop = select_diverse(pop, 10)
    println("New group size: ", length(pop))
end
