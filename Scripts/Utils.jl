using CSV
using DataFrames

df = CSV.File("min_max_adj.csv") |> DataFrame!
const MINS = Array{Float64}(df[!, :min])
const MAXS = Array{Float64}(df[!, :adj_max])
const DIFF = MAXS .- MINS
const ON_INP = ~.(Bool.(df[!, :off]))
# we import the list of our features into a dict to map array indexes
const FEATURES_MAP = JSON.parsefile("features_list2.json")
const TIME_TO_KILL = Dict(300.0=>1, 500.0=>2, 700.0=>3)

"""
Helper functions to get the fitness whatever agent you are using.
"""
function Fitness(lastState::Array{Float64}{1})
	# array index in Julia start at 1
	gold = lastState[FEATURES_MAP["gold"]+1]
	return gold
end

function Fitness1(lastState::Array{Float64}{1}, nbKill::Int64, nbDeath::Int64, earlyPenalty::Int64)
    netWorth = lastState[FEATURES_MAP["net worth"]+1]
    level = lastState[FEATURES_MAP["level"]+1]
    lastHits = lastState[FEATURES_MAP["last hits"]+1]
    denies = lastState[FEATURES_MAP["denies"]+1]
    towerHealth = lastState[FEATURES_MAP["bad tower health"]+1]
    maxTowerHealth = lastState[FEATURES_MAP["bad tower max health"]+1]
    ratioTower = (maxTowerHealth-towerHealth)/maxTowerHealth
	  reward = max(100*lastHits + 100*denies + 2000*ratioTower + 1000*nbKill - 1000*earlyPenalty,
                 0.0)
    return reward
end

"""
transform features into inputs
"""
function get_inputs(features::Array{Float64})
    inputs = (lastFeatures .- MINS) ./ DIFF
    inputs = min.(max.(inputs, -1.0), 1.0)
    inputs[ON_INP]
end

"""
Helper function to know if you should quit the game earlier than the GameOver
"""
function EarlyStop(lastState::Array{Float64}{1})
	dotaTime = lastState[FEATURES_MAP["dota time"]+1]
	lastHits = lastState[FEATURES_MAP["last hits"]+1]
	for (timeCreep,killedCreep) in TIME_TO_KILL
		if (dotaTime>timeCreep&&lastHits<killedCreep)
			return true
		end
	end
	return false
end

"""
Function to save a CGP Individual
"""
function SaveInd(ind::Individual;name::String="None")
	saveFile = JSON.json(String(ind))
	if (name=="None")
		fileName = Dates.now()
		fileName = Dates.format(fileName, "yyyy-mm-dd-HH-MM")
	else
		fileName = name
	end
	open("best/$fileName.json","w") do f
		write(f,saveFile)
	end
end

"""
Function to load a CGP Individual
"""
function LoadInd(path::String)
	indInfo = JSON.parsefile(path)
	return CGPInd(cfg,indInfo)
end

"""
Function to load an entire population from a gen folder.
The evolution need to be initialized first.
"""
function LoadGen(e::Evolution, path::String)
	individualNameList = readdir("gens/$path")
	individualList = Cambrian.Individual[]
	for i in eachindex(individualNameList)
		indString = read("gens/$path/$(individualNameList[i])", String)
		ind = CGPInd(cfg,indString)
		push!(individualList,ind)
	end
	e.population = individualList
end

"""
Function to change the evolution id
"""
function ChangeId(e::Evolution;name::String="None")
	if (name == "None")
		id = Dates.now()
		id = Dates.format(id, "dd-mm-yyyy-HH-MM")
		id = "GenerationsFrom-$id"
	else
		id = name
	end
	e.id = id
end

"""
Helper function to estimate the damage made to the opponent champion between two state
"""
function EstimateDamage(oldLastState::Array{Float64},lastState::Array{Float64})
	if length(oldLastState) != 1
		deltaTime = lastState[FEATURES_MAP["dota time"]+1] - oldLastState[FEATURES_MAP["dota time"]+1]
		if ((lastState[FEATURES_MAP["dota time"]+1] - lastState[FEATURES_MAP["last attack time"]+1])<=deltaTime)
			healthOppRegen = lastState[FEATURES_MAP["opp health regen"]+1]
			oppHealthCurrent = lastState[FEATURES_MAP["opp health"]+1]
			oppHealthOld = oldLastState[FEATURES_MAP["opp health"]+1]
			damage = max(0,oppHealthOld + deltaTime*healthOppRegen - oppHealthCurrent)
			return damage
		end
	end
	return 0
end

"""
Function to get the final health ratio of the opponent tower
"""
function GetTowerRatio(lastState::Array{Float64})
	towerHealth = lastState[FEATURES_MAP["bad tower health"]+1]
	maxTowerHealth = lastState[FEATURES_MAP["bad tower max health"]+1]
	ratioTower = (maxTowerHealth-towerHealth)/maxTowerHealth
	return ratioTower
end

"""
Helper functions to load and save map
"""
function save_map(map_el::MAPElites.MapElites,path::String;name::String="None")
	mkpath(path)
	dict = Dict("feature_dimension"=>mapel.feature_dimension,
       "grid_mesh"=>mapel.grid_mesh,
       "solutions"=>Dict((key[1]-1)*mapel.grid_mesh+(key[2]-1) => String(value) for (key,value) in mapel.solutions.data),
       "performances"=>Dict((key[1]-1)*mapel.grid_mesh+(key[2]-1) => value for (key,value) in mapel.performances.data)
    )
	saveFile = JSON.json(dict)
	if (name=="None")
		fileName = Dates.now()
		fileName = Dates.format(fileName, "yyyy-mm-dd-HH-MM")
	else
		fileName = name
	end
	open("$path/$fileName.json","w") do f
		write(f,saveFile)
	end
end

function load_map(path::String)
	mapInfo = JSON.parsefile(path)
	feature_dimension = mapInfo["feature_dimension"]
	grid_mesh = mapInfo["grid_mesh"]

	mapel = MAPElites.MapElites(feature_dimension,grid_mesh)

	solutions_data = mapInfo["solutions"]
	solutions_data = Dict(parse(Int,key) =>CGPInd(cfg,value) for (key,value) in solutions_data)
	solutions_data = Dict(intToTuple(key,grid_mesh) => value for (key,value) in solutions_data)

    performances_data = mapInfo["performances"]
    performances_data = Dict(parse(Int,key) => convert(Array{Float64},value) for (key,value) in performances_data)
	performances_data = Dict(intToTuple(key,grid_mesh) => value for (key,value) in performances_data)

	for (key,value) in solutions_data
		mapel.solutions.data[key] = value
	end

	for (key,value) in performances_data
		mapel.performances.data[key] = value
	end
	return mapel
end

"""
This function take into consideration that we have a 2-dim
"""
function intToTuple(int::Int64,grid_mesh::Int64)
	x = 1 + div(int,grid_mesh)
	y = 1 + rem(int,grid_mesh)
	return (x,y)
end
